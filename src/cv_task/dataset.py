import os
import torch
import torchvision
import numpy as np
import pandas as pd
from PIL import Image
import torch.nn as nn
from skimage.io import imread
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split

class LeafImageDataset(Dataset):
    def __init__(self, image_ids, labels, dir_images, image_size=320, is_train_set=True):
        self.image_ids = image_ids
        self.labels = labels
        self.image_size = image_size
        self.dir_images = dir_images
        self.transform = None
        self.is_train_set = is_train_set

        if self.is_train_set:
            self.transform = transforms.Compose([
                transforms.ToPILImage(),
                transforms.Resize((self.image_size, self.image_size), Image.BILINEAR),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225],
                ),
                transforms.RandomHorizontalFlip(),
                transforms.RandomVerticalFlip(),
            ])
        else:
            self.transform = transforms.Compose([
                transforms.ToPILImage(),
                transforms.Resize((self.image_size, self.image_size), Image.BILINEAR),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225],
                ),
            ])

    def __len__(self):
        return len(self.image_ids)

    def __getitem__(self, idx):
        file_image = os.path.join(self.dir_images, self.image_ids[idx])
        image = imread(file_image)
        image = self.transform(image)
        label = self.labels[idx]
        return image, label

def split_dataset(file_csv, is_for_train=True, random_state=4):
    data_frame = pd.read_csv(file_csv)
    num_classes = len(np.unique(data_frame["label"].to_numpy()))

    train_x, test_x, train_y, test_y = train_test_split(data_frame["image_id"].to_numpy(), data_frame["label"].to_numpy(), test_size=0.2, random_state=random_state)
    train_x, valid_x, train_y, valid_y = train_test_split(train_x, train_y, test_size=0.1, random_state=random_state)

    if is_for_train:
        return train_x, valid_x, train_y, valid_y, num_classes
    else:
        return test_x, test_y, num_classes

def get_dataloaders_for_training(train_x, train_y, valid_x, valid_y, dir_images, image_size=320, batch_size=8):
    train_dataset = LeafImageDataset(train_x, train_y, dir_images=dir_images, image_size=image_size, is_train_set=True)
    valid_dataset = LeafImageDataset(valid_x, valid_y, dir_images=dir_images, image_size=image_size, is_train_set=False)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
    )
    valid_loader = DataLoader(
        valid_dataset,
        batch_size=batch_size,
        shuffle=False,
    )
    return train_loader, valid_loader
