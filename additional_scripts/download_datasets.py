import argparse
import os
import subprocess
import zipfile
import shutil
from pathlib import Path

import wget

LOCAL_DATA_DIR = Path('local_data')
BOP_DS_DIR = LOCAL_DATA_DIR / 'bop_datasets'
RCLONE_ROOT = 'cosypose:'
DOWNLOAD_DIR = LOCAL_DATA_DIR / 'downloads'
DOWNLOAD_DIR.mkdir(exist_ok=True, parents=True)

BOP_SRC = 'https://bop.felk.cvut.cz/media/data/bop_datasets/'
BOP_DATASETS = {
    'ycbv': {
        'splits': ['train_real', 'train_synt', 'test_all']
    },

    'tless': {
        'splits': ['test_primesense_all', 'train_primesense'],
    },

    'hb': {
        'splits': ['test_primesense_all', 'val_primesense'],
    },

    'icbin': {
        'splits': ['test_all'],
    },

    'itodd': {
        'splits': ['val', 'test_all'],
    },

    'lm': {
        'splits': ['test_all'],
    },

    'lmo': {
        'splits': ['test_all'],
        'has_pbr': False,
    },

    'tudl': {
        'splits': ['test_all', 'train_real']
    },
}

BOP_DS_NAMES = list(BOP_DATASETS.keys())


def main():
    parser = argparse.ArgumentParser('CosyPose download utility')
    parser.add_argument('--models', default='', type=str, choices=BOP_DS_NAMES)
    parser.add_argument('--bop_dataset', default='', type=str, choices=BOP_DS_NAMES)
    parser.add_argument('--augmentation_textures', action='store_true')
    parser.add_argument('--test_datasets_only', action='store_true')
    args = parser.parse_args()

    if args.models:
        download_bop_models(args.models)

    if args.bop_dataset:
        download_bop_original(args.bop_dataset, BOP_DATASETS[args.bop_dataset].get('has_pbr', True), args.test_datasets_only)

    if args.augmentation_textures:
        tmp_path = DOWNLOAD_DIR / "VOCtrainval_11-May-2012.tar"
        wget_file("http://host.robots.ox.ac.uk/pascal/VOC/voc2012/VOCtrainval_11-May-2012.tar", tmp_path)
        print('\nExtracting textures...')
        subprocess.run(["tar", "-xf", tmp_path])
        shutil.move("VOCdevkit", LOCAL_DATA_DIR / "VOCdevkit")

def download_bop_original(ds_name, download_pbr, test_only=False):
    filename = f'{ds_name}_base.zip'
    wget_download_and_extract(BOP_SRC + filename, BOP_DS_DIR)

    suffixes = ['models'] + BOP_DATASETS[ds_name]['splits']
    if download_pbr:
        suffixes += ['train_pbr']
    if test_only:
        suffixes = [f for f in suffixes if (("train" not in f) and ("val" not in f))]
    for suffix in suffixes:
        wget_download_and_extract(BOP_SRC + f'{ds_name}_{suffix}.zip', BOP_DS_DIR / ds_name)

def download_bop_models(ds_name):
    wget_download_and_extract(BOP_SRC + f'{ds_name}_models.zip', BOP_DS_DIR / ds_name)
    if not (BOP_DS_DIR / ds_name / 'models').exists():
        os.symlink(BOP_DS_DIR / ds_name / 'models_eval', BOP_DS_DIR / ds_name / 'models')

def wget_file(url, tmp_path):
    if tmp_path.exists():
        print(f'{url} already downloaded: {tmp_path}...')
    else:
        print(f'Download {url} at {tmp_path}...')
        wget.download(url, out=tmp_path.as_posix())

def wget_download_and_extract(url, out):
    tmp_path = DOWNLOAD_DIR / url.split('/')[-1]
    wget_file(url, tmp_path)
    print(f'\nExtracting {tmp_path} at {out}.')
    zipfile.ZipFile(tmp_path).extractall(out)


if __name__ == '__main__':
    main()
