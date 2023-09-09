# https://github.com/owkin/FLamby/blob/main/flamby/datasets/fed_isic2019/README.md
# resize_images.py with center crop

# https://github.com/owkin/FLamby/blob/main/flamby/datasets/fed_isic2019/dataset.py

import os
import random

import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset


class FedIsic2019(torch.utils.data.Dataset):

    def __init__(self, center=0, split='test', transform=None, data_path=None, val_rate=0.1):
        assert center in list(range(6)) + [-1]
        assert split in ["train", "test", "val"]

        self.center = center
        self.split = split
        self.data_path = data_path
        self.transform = transform

        self.train_test_split_path = os.path.join(self.data_path, 'train_test_split')
        self.image_dir = os.path.join(self.data_path, 'ISIC_2019_Training_Input_preprocessed')

        # Read train_test_split
        df = pd.read_csv(self.train_test_split_path)

        if self.center == -1:
            if self.split in ['train', 'val']:
                key = 'train'
                df2 = df.query("fold == '" + key + "' ").reset_index(drop=True)
            else:
                key = 'test'
                df2 = df.query("fold == '" + key + "' ").reset_index(drop=True)
        else:
            if self.split == 'train':
                key = f'train_{self.center}'
                df2 = df.query("fold2 == '" + key + "' ").reset_index(drop=True)
            elif self.split == 'val':
                key = f'train_{self.center}'
                df2 = df.query("fold2 == '" + key + "' ").reset_index(drop=True)
            else:
                key = 'test'
                df2 = df.query("fold == '" + key + "' ").reset_index(drop=True)

        images, targets = df2.image.tolist(), df2.target.tolist()  # always same order
        samples = [(os.path.join(self.image_dir, image_name + ".jpg"), target) for image_name, target in zip(images, targets)]

        # shuffle with fixed seed
        random.Random(1).shuffle(samples)
        if self.center == -1:
            self.samples = samples  # val handles with user_groups
        else:
            if self.split == 'train':
                self.samples = samples[int(len(samples) * val_rate):]
            elif self.split == 'val':
                self.samples = samples[:int(len(samples) * val_rate)]
            elif self.split == 'test':
                self.samples = samples
            else:
                raise ValueError('')

        self.targets = [s[1] for s in self.samples]

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        image_path, target = self.samples[idx]
        image = Image.open(image_path).convert('RGB')
        image = self.transform(image)
        return image, target
