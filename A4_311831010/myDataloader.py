# %%
import datetime
import os
import time
from pathlib import Path
from typing import Dict, List, Optional, Union

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset
from torchvision import utils as vutils


class EMnistDataset(Dataset):
    def __init__(self, img_dir, label_dir="", transform=None):
        self.img_dir = img_dir
        self.label_dir = label_dir
        self.transform = transform
        self.imgs = np.load(self.img_dir, allow_pickle=True)
        if self.label_dir != "":
            self.labels = np.load(self.label_dir, allow_pickle=True)
            print(f'label len: {self.labels.shape}')
        print(f'img len: {self.imgs.shape}')
    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, idx):

        image = self.imgs[idx]
        #label = self.data[1][idx]
        if self.transform:
            image = self.transform(image)
        if self.label_dir != "":
            return image, self.labels[idx]
        else:
            return image