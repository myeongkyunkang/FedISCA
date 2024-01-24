# FedISCA
MICCAI2023. "One-shot Federated Learning on Medical Data using Knowledge Distillation with Image Synthesis and Client Model Adaptation"

# Train Classifiers

    # bloodmnist, dermamnist, octmnist, pathmnist, tissuemnist
    CUDA_VISIBLE_DEVICES=0 python train_classifier.py \
        --root ./dataset \
        --output_dir ./pretrained_models \
        --dataset bloodmnist --partition iid

    # rsna
    CUDA_VISIBLE_DEVICES=0 python train_classifier.py  \
        --root ./dataset \
        --output_dir ./pretrained_models \
        --dataset rsna --aug --pretrained --partition iid

    # diabetic2015
    CUDA_VISIBLE_DEVICES=0 python train_classifier.py  \
        --root ./dataset \
        --output_dir ./pretrained_models \
        --dataset diabetic2015 --aug --pretrained --partition iid

    # isic2019_merge
    CUDA_VISIBLE_DEVICES=0 python train_classifier.py  \
        --root ./dataset \
        --output_dir ./pretrained_models \
        --dataset isic2019_merge --aug --pretrained --partition iid

    # isic2019
    CUDA_VISIBLE_DEVICES=0 python train_classifier.py  \
        --root ./dataset \
        --output_dir ./pretrained_models \
        --dataset isic2019 --aug --pretrained


# Run FedISCA

```
python
import os
GPU='0'
dataset_tag_list = ['iid_5_0.6']
dataset_list = ['bloodmnist']  # 'bloodmnist', 'dermamnist', 'octmnist', 'pathmnist', 'tissuemnist'
for dataset_tag in dataset_tag_list:
    for dataset in dataset_list:
        os.system(f"CUDA_VISIBLE_DEVICES={GPU} python main_fedisca.py \
            --dataset {dataset} \
            --root ./dataset \
            --teacher_weights=./pretrained_models/{dataset}_{dataset_tag} \
            --exp_descr=./results/oneshot_{dataset_tag}/{dataset}")

python
import os
GPU='0'
dataset_list = ['rsna']  # 'rsna', 'diabetic2015'
dataset_tag_list = ['iid_5_0.6']
for dataset_tag in dataset_tag_list:
    for dataset in dataset_list:
        os.system(f"CUDA_VISIBLE_DEVICES={GPU} python main_fedisca.py \
            --dataset {dataset} \
            --root ./dataset \
            --teacher_weights=./pretrained_models/{dataset}_{dataset_tag} --pretrained \
            --exp_descr=./results/oneshot_{dataset_tag}/{dataset}_pretrained --bs 50 --iters_mi 1000")

python
import os
GPU='0'
dataset_list = ['isic2019']
dataset_tag_list = ['merge_iid_5_0.6']  # 'merge_iid_5_0.6', 'dirichlet_6_0.6'
for dataset_tag in dataset_tag_list:
    for dataset in dataset_list:
        os.system(f"CUDA_VISIBLE_DEVICES={GPU} python main_fedisca.py \
            --dataset {dataset} \
            --root ./dataset \
            --teacher_weights=./pretrained_models/{dataset}_{dataset_tag} --pretrained \
            --exp_descr=./results/oneshot_{dataset_tag}/{dataset}_pretrained --bs 50 --iters_mi 1000")
```


# Datasets

    # small-scale datasets
    https://medmnist.com/
    # download bloodmnist, dermamnist, octmnist, pathmnist, tissuemnist npz files and move to the ./datasets/medmnist directory

    # large-scale datasets
    https://www.kaggle.com/c/rsna-pneumonia-detection-challenge
    https://www.kaggle.com/datasets/benjaminwarner/resized-2015-2019-blindness-detection-images
    https://challenge.isic-archive.com/landing/2019/
    https://github.com/owkin/FLamby/blob/main/flamby/datasets/fed_isic2019/README.md
    # check tools for preprocessing


# Environments

    pip install -U pip
    pip install -r requirements.txt
