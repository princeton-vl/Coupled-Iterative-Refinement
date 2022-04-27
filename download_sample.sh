wget https://www.dropbox.com/s/66d7stwhkgilkl9/sample_ycbv.zip
unzip sample_ycbv.zip
wget https://www.dropbox.com/s/fknurzz0x848ltj/sample_tless.zip
unzip sample_tless.zip
wget https://www.dropbox.com/s/mr0cqzzzfvnawvo/sample_lmo.zip
unzip sample_lmo.zip

mkdir -p local_data/bop_datasets/ycbv
cd local_data/bop_datasets/ycbv
wget https://www.dropbox.com/s/sipsptaasn9xynt/ycbv_obj_models.zip
unzip ycbv_obj_models.zip
cd ../../..

mkdir -p local_data/bop_datasets/tless
cd local_data/bop_datasets/tless
wget https://www.dropbox.com/s/908wy7ht1bzwwbe/tless_obj_models.zip
unzip tless_obj_models.zip
cd ../../..

mkdir -p local_data/bop_datasets/lm
cd local_data/bop_datasets/lm
wget https://www.dropbox.com/s/4l1udyk0cz28alv/lm_obj_models.zip
unzip lm_obj_models.zip
cd ../../..