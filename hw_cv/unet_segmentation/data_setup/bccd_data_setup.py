import os

import cv2
import pandas as pd
import torch
import torch.utils.data
import torchvision as tv
from torch.utils.data import DataLoader
import random


class BCCDDataset(torch.utils.data.Dataset):
    def __init__(self, files, device, root='/home/alex/projects/ml/ml_hw/hw_cv/bccd-dataset/'):
        super()
        self.root = root
        self.transforms = tv.transforms.Compose([
            # tv.transforms.Resize([300, 300]),
            tv.transforms.ToTensor(),
            tv.transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])
        self.imgs = files
        self.device = device

    def __getitem__(self, idx):
        filename = self.imgs[idx]
        img = cv2.imread(os.path.join(self.root + 'BCCD/JPEGImages', filename))
        gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        ret, threshold = cv2.threshold(gray_img, 190, 255, cv2.THRESH_BINARY)
        threshold = threshold / 255.0

        img = cv2.resize(img, (224, 224))
        threshold = cv2.resize(threshold, (224, 224))

        img = self.transforms(img).to(self.device)

        return torch.Tensor(img).float().to(self.device), torch.Tensor(threshold).long().to(self.device)

    def __len__(self):
        return len(self.imgs)


def split_files(files_folder):
    images_dir = os.path.join(files_folder)
    files_in_folder = os.listdir(images_dir)
    random.shuffle(files_in_folder)
    split_point = int(0.8 * len(files_in_folder))

    training_files = files_in_folder[:split_point]
    testing_files = files_in_folder[split_point:]

    return training_files, testing_files


def create_dataloaders(batch_size, bccd_dataset_location, device):
    training_files, testing_files = split_files(os.path.join(bccd_dataset_location, 'BCCD/JPEGImages'))

    trn_ds = BCCDDataset(files=training_files, root=bccd_dataset_location, device=device)
    val_ds = BCCDDataset(files=testing_files, root=bccd_dataset_location, device=device)
    trn_dl = DataLoader(trn_ds, batch_size=batch_size, shuffle=True)
    val_dl = DataLoader(val_ds, batch_size=batch_size, shuffle=True)

    return trn_dl, val_dl
