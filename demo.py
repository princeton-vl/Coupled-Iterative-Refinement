import argparse
import json
from pathlib import Path
from test import generate_pose_from_detections
from tqdm import tqdm

import colored_traceback
import gin
import imageio
import numpy as np
import torch
from lietorch import SE3

from crops import crop_inputs
from detector import load_detector
from pose_models import load_efficientnet
from train import format_gin_override, load_raft_model
from utils import Pytorch3DRenderer, get_perturbations
from datasets import lin_interp

torch.backends.cudnn.deterministic = False
torch.backends.cudnn.benchmark = True

def read_depth(depth_path: Path, depth_scale: float, interpolate=False):
    depth = np.array(imageio.imread(depth_path).astype(np.float32))
    depth = depth * depth_scale / 1000
    if interpolate: # interpolating the missing depth values takes about 0.7s, scipy is slow
        return lin_interp(depth)
    return depth

@torch.no_grad()
def main():
    colored_traceback.add_hook()
    parser = argparse.ArgumentParser()
    parser.add_argument('--scene_dir', type=Path, required=True, help="A folder with an rgb/ subdir, a scene_camera.json, and (Optionally) a depth/ subdir")
    parser.add_argument('--output_dir', type=Path, default="qualitative_output", help="The directory to save qualitative output")
    parser.add_argument('-o', '--override', nargs='+', type=str, default=[], help="gin-config settings to override")
    parser.add_argument('--load_weights', type=str, required=True, help='path to the model weights to load')
    parser.add_argument('--num_outer_loops', type=int, default=2, help="number of outer-loops in each forward pass")
    parser.add_argument('--num_inner_loops', type=int, default=10, help="number of inner-loops in each forward pass")
    parser.add_argument('--num_solver_steps', type=int, default=3, help="number of BD-PnP solver steps per inner-loop (doesn't affect Modified BD-PnP)")
    parser.add_argument('--obj_models', required=True, choices=['ycbv', 'tless', 'lmo', 'hb', 'tudl', 'icbin', 'itodd'], help="which object models to use")
    parser.add_argument('--rgb_only', action='store_true', help="use the RGB-only model")
    args = parser.parse_args()
    args.override = format_gin_override(args.override)
    gin.parse_config_files_and_bindings(["configs/base.gin", f"configs/test_{args.obj_models}_{'rgb' if args.rgb_only else 'rgbd'}.gin"], args.override)

    detector = load_detector()

    run_efficientnet = load_efficientnet()

    model = load_raft_model(args.load_weights)
    model.eval()
    Pytorch3DRenderer() # Loading Renders. This gets cached so it's only slow the first time.

    print(f"\n\nSaving images output to {args.output_dir}/\n\n")
    args.output_dir.mkdir(exist_ok=True)

    if not args.scene_dir.exists():
        raise FileNotFoundError(f"The directory {args.scene_dir} doesn't exist. Download a sample scene using ./download_sample.sh or set --scene_dir to a BOP scene directory.")
    if not (args.scene_dir / "rgb").exists():
        raise FileNotFoundError(f"The directory {args.scene_dir / 'rgb'} doesn't exist.")
    if not (args.scene_dir / "scene_camera.json").exists():
        raise FileNotFoundError(f"The file {args.scene_dir / 'scene_camera.json'} doesn't exist.")
    if not args.rgb_only and not (args.scene_dir / "depth").exists():
        raise FileNotFoundError(f"The directory {args.scene_dir / 'depth'} doesn't exist, and --rgb_only isn't used.")

    scene_cameras = json.loads((args.scene_dir / "scene_camera.json").read_text())
    image_loop = list(scene_cameras.items())
    np.random.default_rng(0).shuffle(image_loop)

    for image_index, (frame_id, scene_camera) in enumerate(image_loop):
        camera_intrinsics = torch.as_tensor(scene_camera['cam_K'], device='cuda', dtype=torch.float32).view(1,3,3)
        depth_scale = scene_camera['depth_scale']
        rgb_path = args.scene_dir / "rgb" / f"{int(frame_id):06d}.png"
        images = imageio.imread(rgb_path)
        render_resolution = torch.tensor(images.shape[:2], device='cuda', dtype=torch.float32).view(1,2) / 2
        images = torch.as_tensor(images, device='cuda', dtype=torch.float32).permute(2,0,1).unsqueeze(0) / 255
        if args.rgb_only:
            interpolated_depth = torch.zeros_like(images[:,0])
        else:
            depth_path = args.scene_dir / "depth" / rgb_path.name
            interpolated_depth = read_depth(depth_path, depth_scale, interpolate=True)
            interpolated_depth = torch.as_tensor(interpolated_depth, device='cuda', dtype=torch.float32).unsqueeze(0)

        # Generate candidate detections using a Mask-RCNN
        detections = detector.get_detections(images=images)

        # Convert the predicted bounding boxes to initial translation estimates
        data_TCO_init = generate_pose_from_detections(detections=detections, K=camera_intrinsics)

        for obj_idx, (_, obj_label, _) in tqdm(list(detections.infos.iterrows()), desc=f"Predicting poses for {len(detections)} detected objects in image {image_index+1}/{len(image_loop)}"):
            mrcnn_mask = detections.masks[[obj_idx]]
            mrcnn_pose = data_TCO_init.poses[[obj_idx]]
            basename = f"{image_index}.{obj_idx+1}"

            # Crop the image given the translation predicted by the Mask-RCNN
            images_cropped, K_cropped, _, _, masks_cropped, depths_cropped = crop_inputs(images=images, K=camera_intrinsics, TCO=mrcnn_pose, \
                labels=[obj_label], masks=mrcnn_mask, sce_depth=interpolated_depth, render_size=render_resolution.squeeze().cpu().numpy())

            mrcnn_rendered_rgb, _, _ = Pytorch3DRenderer()([obj_label], mrcnn_pose, K_cropped, render_resolution)
            imageio.imwrite(args.output_dir / f"{basename}_1_Mask_RCNN_Initial_Translation.png", mrcnn_rendered_rgb[0].permute(1,2,0).mul(255).byte().cpu())

            # Generate a coarse pose estimate using an efficientnet
            assert (mrcnn_rendered_rgb.shape == images_cropped.shape)
            images_input = torch.cat((images_cropped, mrcnn_rendered_rgb), dim=1)
            current_pose_est = run_efficientnet(images_input, mrcnn_pose, K_cropped)

            efficientnet_rendered_rgb, _, _ = Pytorch3DRenderer()([obj_label], current_pose_est, K_cropped, render_resolution)
            imageio.imwrite(args.output_dir / f"{basename}_2_Efficientnet_Prediction.png", efficientnet_rendered_rgb[0].permute(1,2,0).mul(255).byte().cpu())

            for outer_loop_idx in range(args.num_outer_loops):
                # Crop image given the previous pose estimate
                images_cropped, K_cropped, _, _, masks_cropped, depths_cropped = crop_inputs(images=images, K=camera_intrinsics, TCO=current_pose_est, \
                labels=[obj_label], masks=mrcnn_mask, sce_depth=interpolated_depth, render_size=render_resolution.squeeze().cpu().numpy())

                # Render additional viewpoints
                input_pose_multiview = get_perturbations(current_pose_est).flatten(0,1)
                Nr = input_pose_multiview.shape[0]
                label_rep = np.repeat([obj_label], Nr)
                K_rep = K_cropped.repeat_interleave(Nr, dim=0)
                res_rep = render_resolution.repeat_interleave(Nr, dim=0)
                rendered_rgb, rendered_depth, _ = Pytorch3DRenderer()(label_rep, input_pose_multiview, K_rep, res_rep)
                
                # Forward pass
                combine = lambda a, b: torch.cat((a.unflatten(0, (1, Nr)), b.unsqueeze(1)), dim=1)
                images_input = combine(rendered_rgb, images_cropped)
                depths_input = combine(rendered_depth, depths_cropped)
                masks_input = combine(rendered_depth > 1e-3, masks_cropped)
                pose_input = combine(input_pose_multiview, current_pose_est)
                K_input = combine(K_rep, K_cropped)

                outputs = model(Gs=pose_input, images=images_input, depths_fullres=depths_input, \
                    masks_fullres=masks_input, intrinsics_mat=K_input, labels=[obj_label], \
                        num_solver_steps=args.num_solver_steps, num_inner_loops=args.num_inner_loops)
                current_pose_est = SE3(outputs['Gs'][-1].contiguous()[:, -1]).matrix()

                efficientnet_rendered_rgb, _, _ = Pytorch3DRenderer()([obj_label], current_pose_est, K_cropped, render_resolution)
                imageio.imwrite(args.output_dir / f"{basename}_3_CIR_Outer-Loop-{outer_loop_idx}.png", efficientnet_rendered_rgb[0].permute(1,2,0).mul(255).byte().cpu())

            imageio.imwrite(args.output_dir / f"{basename}_4_Image_Crop.png", images_cropped[0].permute(1,2,0).mul(255).byte().cpu())


if __name__ == '__main__':
    main()
