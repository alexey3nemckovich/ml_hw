import cv2
import numpy as np
import torch
import torch.utils.data
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from .circles_data_provider import RgbDataProvider


class SegData(Dataset):
    def __init__(self, generator, size, device):
        self.images, self.masks = generator(size)
        self.transforms = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(
                [0.485, 0.456, 0.406],
                [0.229, 0.224, 0.225])
        ])
        self.device = device

    def __len__(self):
        return len(self.images)

    def __getitem__(self, ix):
        """
        В данном методе мы будем ресайзить как картинку (наш инпут) так и маску (аутпут) для того
        чтобы они имели одинаковую размерность и наша архитектура была способно обработать 2 изображения
        одновременно без ошибок.
        Маска содержит целочисленные значения которые лежат в промежутке от 0 до 11 (то есть у нас всего 12 потенциальных классов
        в размеченной маске).
        """
        image = self.images[ix]
        mask = self.masks[ix][..., 1]
        image = cv2.resize(image, (224, 224))
        mask = cv2.resize(mask, (224, 224))
        return image, mask

    def choose(self):
        return self[np.random.randint(len(self))]

    def collate_fn(self, batch):
        """
        Метод который обрабатывает наш батч (читает маску и картинку, преваращет в тензор, нормализирует изображение (не маску!!!!) и отправляет на указанный девайс.
        """
        ims, masks = list(zip(*batch))
        ims = torch.cat([self.transforms(im.copy())[None] for im in ims]).float().to(self.device)
        ce_masks = torch.cat([torch.Tensor(mask[None]) for mask in masks]).long().to(self.device)
        return ims, ce_masks


def create_dataloaders(batch_size, train_data_size, validation_data_size, device):
    nx = 572
    ny = 572

    generator = RgbDataProvider(nx, ny, cnt=20)

    trn_ds = SegData(generator, train_data_size, device)
    val_ds = SegData(generator, validation_data_size, device)
    trn_dl = DataLoader(trn_ds, batch_size=batch_size, shuffle=True, collate_fn=trn_ds.collate_fn)
    val_dl = DataLoader(val_ds, batch_size=batch_size, shuffle=True, collate_fn=val_ds.collate_fn)

    return trn_dl, val_dl
