import os
import time
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.nn import BCEWithLogitsLoss
from torch.optim import Adam
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms as xfms
from pathlib import Path
from torch import Tensor

from tqdm import tqdm
from PIL import Image
import itertools
import cv2
from torchvision.utils import make_grid
import albumentations as A
import albumentations.pytorch.transforms as Atf


import torch.nn as nn
import torch.nn.functional as F

torch.cuda.set_device(0)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("device", torch.cuda.current_device(), torch.cuda.get_device_name(torch.cuda.current_device()))

root = r'dataset'

class MyDataset(Dataset):
    def __init__(self, root_dir, split, first_transform=None, second_transform = None):
        self.img_dir = root_dir+'/images'
        self.mask_dir = root_dir+'/masks'
        self.first_transform = first_transform
        self.second_transform = second_transform
        self.img_path_dic = {}
        # self.mask_path_list = []
        self.ratio = 255//7
        self.split = split

        for class_dir in os.listdir(self.img_dir):
            image_dir_path = os.path.join(self.img_dir, class_dir)
            mask_dir_path = os.path.join(self.mask_dir, class_dir)

            for img_file in os.listdir(image_dir_path):

                name = img_file.split('.')[0]

                image_path = os.path.join(image_dir_path, f'{name}.jpg')

                mask_path = os.path.join(mask_dir_path, f'{name}.png')
                self.img_path_dic[image_path] = mask_path

        shuffle_dic = list(self.img_path_dic.items())
        np.random.shuffle(shuffle_dic)
        self.img_path_dic = dict(shuffle_dic)

        self.train_set = dict(itertools.islice(self.img_path_dic.items(), len(self.img_path_dic)*7//10))
        self.test_set = dict(itertools.islice(self.img_path_dic.items(), len(self.img_path_dic)*7//10, len(self.img_path_dic)))

    def __len__(self):
        if self.split == 'train':
            return len(self.train_set)
        else:
            return len(self.test_set)

    def __getitem__(self, idx):

        if self.split == 'train':
            dataset = self.train_set
        else:
            dataset = self.test_set

        image_path = list(dataset)[idx]
        img = np.array(Image.open(image_path))
        mask_path = list(dataset.values())[idx]
        mask = Image.open(mask_path)
        mask = np.array(mask)#*self.ratio

        # print(image_path)
        # print(mask_path)
        if self.first_transform:
            transformed = self.first_transform(image = img, mask = mask)
            img = transformed["image"]
            mask = transformed["mask"]

        if self.second_transform:
            img = self.second_transform(img.to(torch.float64))#.to(torch.int64)
        return img, mask

split = ['train', 'test']
IMG_SIZE = [800, 800]
STD_MIN = [3513.6, 2428.6]
BATCH_SIZE = 16
INIT_LR = 0.001
EPOCHES = 3
amp = False

tds = MyDataset(root,
                split[0],
                A.Compose([
                    A.RandomCrop(width=IMG_SIZE[1], height=IMG_SIZE[0]),
                    A.Rotate(limit=360, p=0.9, border_mode=cv2.BORDER_CONSTANT),
                    A.HorizontalFlip(p=0.5),
                    Atf.ToTensorV2(),
                ],
                    additional_targets={'image0': 'image', 'mask': 'image'}),
                xfms.Compose([xfms.Normalize(STD_MIN[1], STD_MIN[0])
                              ])
                )

ttds = MyDataset(root,
                 split[1],
                 A.Compose([A.RandomCrop(width=IMG_SIZE[1], height=IMG_SIZE[0]),
                            Atf.ToTensorV2(),
                            ],
                           additional_targets={'image0': 'image', 'mask': 'image'}),
                 xfms.Compose([xfms.Normalize(STD_MIN[1], STD_MIN[0])
                               ])
                 )

tdl = DataLoader(tds, batch_size=BATCH_SIZE, shuffle=True)
vdl = DataLoader(ttds, batch_size=BATCH_SIZE, shuffle=True)

def dice_coeff(input: Tensor, target: Tensor, reduce_batch_first: bool = False, epsilon: float = 1e-6):
    # Average of Dice coefficient for all batches, or for a single mask
    assert input.size() == target.size()
    assert input.dim() == 3 or not reduce_batch_first

    sum_dim = (-1, -2) if input.dim() == 2 or not reduce_batch_first else (-1, -2, -3)

    inter = 2 * (input * target).sum(dim=sum_dim)
    sets_sum = input.sum(dim=sum_dim) + target.sum(dim=sum_dim)
    sets_sum = torch.where(sets_sum == 0, inter, sets_sum)

    dice = (inter + epsilon) / (sets_sum + epsilon)
    return dice.mean()


def multiclass_dice_coeff(input: Tensor, target: Tensor, reduce_batch_first: bool = False, epsilon: float = 1e-6):
    # Average of Dice coefficient for all classes
    return dice_coeff(input.flatten(0, 1), target.flatten(0, 1), reduce_batch_first, epsilon)


def dice_loss(input: Tensor, target: Tensor, multiclass: bool = False):
    # Dice loss (objective to minimize) between 0 and 1
    fn = multiclass_dice_coeff if multiclass else dice_coeff
    return 1 - fn(input, target, reduce_batch_first=True)

class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)


class Down(nn.Module):
    """Downscaling with maxpool then double conv"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels)
        )

    def forward(self, x):
        return self.maxpool_conv(x)


class Up(nn.Module):
    """Upscaling then double conv"""

    def __init__(self, in_channels, out_channels, bilinear=True):
        super().__init__()

        # if bilinear, use the normal convolutions to reduce the number of channels
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels, in_channels // 2)
        else:
            self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        # input is CHW
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        # if you have padding issues, see
        # https://github.com/HaiyongJiang/U-Net-Pytorch-Unstructured-Buggy/commit/0e854509c2cea854e247a9c615f175f76fbb2e3a
        # https://github.com/xiaopeng-liao/Pytorch-UNet/commit/8ebac70e633bac59fc22bb5195e513d5832fb3bd
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)

class UNet(nn.Module):
    def __init__(self, n_channels, n_classes, bilinear=False):
        super(UNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        self.inc = (DoubleConv(n_channels, 32))
        self.down1 = (Down(32, 64))
        self.down2 = (Down(64, 128))
        self.down3 = (Down(128, 256))
        factor = 2 if bilinear else 1
        #self.down4 = (Down(256, 512 // factor))
        #self.up1 = (Up(512, 256 // factor, bilinear))
        self.up2 = (Up(256, 128 // factor, bilinear))
        self.up3 = (Up(128, 64 // factor, bilinear))
        self.up4 = (Up(64, 32, bilinear))
        self.outc = (OutConv(32, n_classes))

    def forward(self, x):
        x1 = self.inc(x)        # out : 64, 800, 800
        x2 = self.down1(x1)     # out : 128, 400, 400
        x3 = self.down2(x2)     # out : 256, 200, 200
        x4 = self.down3(x3)     # out : 512, 100, 100
        #x5 = self.down4(x4)     # out : 1024, 50, 50
        #x = self.up1(x5, x4)
        x = self.up2(x4, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        logits = self.outc(x)
        return logits

    def use_checkpointing(self):
        self.inc = torch.utils.checkpoint(self.inc)
        self.down1 = torch.utils.checkpoint(self.down1)
        self.down2 = torch.utils.checkpoint(self.down2)
        self.down3 = torch.utils.checkpoint(self.down3)
        self.down4 = torch.utils.checkpoint(self.down4)
        self.up1 = torch.utils.checkpoint(self.up1)
        self.up2 = torch.utils.checkpoint(self.up2)
        self.up3 = torch.utils.checkpoint(self.up3)
        self.up4 = torch.utils.checkpoint(self.up4)
        self.outc = torch.utils.checkpoint(self.outc)

weights_root = Path("weights")
weights_root.mkdir(exist_ok=True)

#unet = UNet(n_channels=3, n_classes=8).to(device)

unet = torch.nn.DataParallel(UNet(n_channels=3, n_classes=8).to(device),
                             device_ids=[0, 1]#, 2, 3, 4, 5, 6, 7]
                             )

opt = Adam(unet.parameters(), lr=INIT_LR)

grad_scaler = torch.cuda.amp.GradScaler(enabled=amp)
criterion = nn.CrossEntropyLoss() if UNet(n_channels=3, n_classes=8).n_classes > 1 else nn.BCEWithLogitsLoss()
global_step = 0
H = {"train_loss": [], "test_loss": []}


startTime = time.time()
for epoch in range(1, EPOCHES + 1):
        unet.train()
        totalTrainLoss = torch.Tensor([0]).to(device)
        totalTestLoss = torch.Tensor([0]).to(device)
        with tqdm(total=len(tds), desc=f'Epoch {epoch}/{EPOCHES}', unit='img') as pbar:
            for batch in tdl:
                images, true_masks = batch[0], batch[1]

                images = images.to(device=device, dtype=torch.float32)
                true_masks = true_masks.to(device=device, dtype=torch.long)

                with torch.autocast(device.type if device.type != 'mps' else 'cpu', enabled=amp):
                    masks_pred = unet(images)
                    loss = criterion(masks_pred, true_masks.squeeze(1))
                    dice_losses = 0
                    for B_i in range(true_masks.shape[0]):
                        # print('pred',F.softmax(masks_pred[B_i], dim=1).shape)
                        # print('true',F.one_hot(true_masks[B_i], 8).permute(0, 3, 1, 2).shape)
                        dice_losses += dice_loss(
                            F.softmax(masks_pred[B_i].unsqueeze(0), dim=1).float(),
                            F.one_hot(true_masks[B_i], 8).permute(0, 3, 1, 2).float(),
                            multiclass=True
                        )
                    loss += dice_losses/true_masks.shape[0]

                opt.zero_grad(set_to_none=True)
                grad_scaler.scale(loss).backward()
                torch.nn.utils.clip_grad_norm_(unet.parameters(), 1)
                grad_scaler.step(opt)
                grad_scaler.update()

                pbar.update(images.shape[0])
                global_step += 1

                totalTrainLoss += loss

                pbar.set_postfix(**{'loss (batch)': loss.item()})

                # Evaluation round
                # division_step = (len(tds) // (5 * BATCH_SIZE))

                # val
                break

            with torch.no_grad():
                unet.eval()
                with tqdm(total=len(tds), desc=f'Epoch {epoch}/{EPOCHES}', unit='img') as pbar:
                    for batch in vdl:

                        images, true_masks = batch[0], batch[1]

                        images = images.to(device=device, dtype=torch.float32)
                        true_masks = true_masks.to(device=device, dtype=torch.long)

                        pred = unet(images)
                        test_loss = criterion(pred, true_masks.squeeze(1))

                        dice_losses = 0
                        for B_i in range(true_masks.shape[0]):
                            print('pred',F.softmax(masks_pred[B_i].unsqueeze(0), dim=1).shape)
                            print('true',F.one_hot(true_masks[B_i], 8).permute(0, 3, 1, 2).shape)
                            dice_losses += dice_loss(
                                F.softmax(masks_pred[B_i].unsqueeze(0), dim=1).float(),
                                F.one_hot(true_masks[B_i], 8).permute(0, 3, 1, 2).float(),
                                multiclass=True
                            )
                        test_loss += dice_losses/true_masks.shape[0]

                        totalTestLoss += test_loss
                        #anno = all_metrics(pred, y, show=False)
                        pbar.update(images.shape[0])
                        pbar.set_postfix(**{'loss (batch)': test_loss.item()})

            # calculate the average training and validation loss
            avgTrainLoss = (totalTrainLoss.cpu().item() / len(tdl))
            avgTestLoss = (totalTestLoss.cpu().item() / len(vdl))
            # if H["test_loss"] and avgTestLoss < np.min(H["test_loss"]):
            torch.save(unet.state_dict(), f"weights/{avgTestLoss:.4f}_{epoch+1:03}.pth")
            # update our training history
            H["train_loss"].append(avgTrainLoss)
            H["test_loss"].append(avgTestLoss)
            # print the model training and validation information
