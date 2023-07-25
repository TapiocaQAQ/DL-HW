import struct
import numpy as np


train_label_filepath = r'./data/EMNIST/emnist-byclass-train-labels-idx1-ubyte'
train_img_filepath = r'./data/EMNIST/emnist-byclass-train-images-idx3-ubyte'
test_label_filepath = r'./data/EMNIST/emnist-byclass-test-labels-idx1-ubyte'
test_img_filepath = r'./data/EMNIST/emnist-byclass-test-images-idx3-ubyte'

def special_label(labels, imgs):
    special = [10, 11, 13, 14, 15, 16, 17, 23, 26, 27, 29]
    label_list = []
    img_list = []

    for label, img in zip(labels, imgs):

        if label in special:
            label_list.append(special.index(label))
            img_list.append(img)
    return label_list, img_list
for label_path, img_path in [(train_label_filepath, train_img_filepath), (test_label_filepath, test_img_filepath)]:
    labels = []
    with open(label_path, 'rb') as file:
        magic, size = struct.unpack(">II", file.read(8))

        if magic != 2049:
            raise ValueError('Magic number mismatch, expected 2049, got {}'.format(magic))

        labels = np.frombuffer(file.read(), dtype=np.uint8)

    with open(img_path, 'rb') as file:
                magic, size, rows, cols = struct.unpack(">IIII", file.read(16))
                if magic != 2051:
                    raise ValueError('Magic number mismatch, expected 2051, got {}'.format(magic))
                image_data = np.frombuffer(file.read(), dtype=np.uint8)
    images = []
    for i in range(size):
        img = np.array(image_data[i*rows*cols : (i+1)*rows*cols]).reshape(cols, rows)
        img = np.transpose(img)
        images.append(img)

    labels, images = special_label(labels, images)

    if label_path == train_label_filepath:
        np.save(r'data/my_train_2_label', labels)
        np.save(r'data/my_train_2_img', images)
    else:
        np.save(r'data/my_test_2_label', labels)
        np.save(r'data/my_test_2_img', images)