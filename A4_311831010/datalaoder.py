import os
import struct
import random
import matplotlib.pyplot as plt
import numpy as np
from os.path import join
from torch.utils.data import Dataset, DataLoader
import array


train_label_filepath = r'./data/EMNIST/emnist-byclass-train-labels-idx1-ubyte'
train_img_filepath = r'./data/EMNIST/emnist-byclass-train-images-idx3-ubyte'
test_label_filepath = r'./data/EMNIST/emnist-byclass-test-labels-idx1-ubyte'
test_img_filepath = r'./data/EMNIST/emnist-byclass-test-images-idx3-ubyte'


class EMnistDataset(Dataset):
    def __init__(self, img_filepath, label_filepath,train=True, transform=None):
        super(EMnistDataset, self).__init__()
        self.img_filepath = img_filepath
        self.label_filepath = label_filepath
        self.transform = transform
        self.train = train
        self.labels, self.images = self.read_imgs_labels(self.img_filepath, self.label_filepath)



    def read_imgs_labels(self, images_path, labels_path):

        labels = []
        with open(labels_path, 'rb') as file:
            magic, size = struct.unpack(">II", file.read(8))

            if magic != 2049:
                raise ValueError('Magic number mismatch, expected 2049, got {}'.format(magic))

            labels = np.frombuffer(file.read(), dtype=np.uint8)
            labels = self.mapping(labels)

        with open(images_path, 'rb') as file:
            magic, size, rows, cols = struct.unpack(">IIII", file.read(16))
            if magic != 2051:
                raise ValueError('Magic number mismatch, expected 2051, got {}'.format(magic))
            image_data = np.frombuffer(file.read(), dtype=np.uint8)
            #[chr(item) for item in image_data]

        images = []
        for i in range(size):
            img = np.array(image_data[i*rows*cols : (i+1)*rows*cols]).reshape(cols, rows)
            images.append(img)

        return labels, images
    def mapping(self,labels):
        chars = []
        for label in labels:
            if label<=10:
               label+=48
            elif label<=35:
                label+=55
            else:
                label += 61
            chars.append(chr(label))
        return chars

    def __getitem__(self, idx):
        if self.train == True:
            image = self.transform(self.images[idx])
        else:
            image = self.transform(self.images[idx])

        return (self.labels[idx], image), (self.labels[idx], image)
    def __len__(self):
        train_label, _ = self.read_imgs_labels(self.img_filepath, self.label_filepath)
        return len(train_label)