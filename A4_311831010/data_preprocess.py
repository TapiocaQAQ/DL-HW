import struct
import numpy as np


train_label_filepath = r'./data/EMNIST/emnist-byclass-train-labels-idx1-ubyte'
train_img_filepath = r'./data/EMNIST/emnist-byclass-train-images-idx3-ubyte'
test_label_filepath = r'./data/EMNIST/emnist-byclass-test-labels-idx1-ubyte'
test_img_filepath = r'./data/EMNIST/emnist-byclass-test-images-idx3-ubyte'

def mapping(labels):
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
for path in (train_img_filepath, test_img_filepath):
    with open(path, 'rb') as file:
                magic, size, rows, cols = struct.unpack(">IIII", file.read(16))
                if magic != 2051:
                    raise ValueError('Magic number mismatch, expected 2051, got {}'.format(magic))
                image_data = np.frombuffer(file.read(), dtype=np.uint8)
    images = []
    for i in range(size):
        img = np.array(image_data[i*rows*cols : (i+1)*rows*cols]).reshape(cols, rows)
        img = np.transpose(img)
        images.append(img)
    if path == train_img_filepath:
        np.save(r'data/my_train_img', images)
    else:
        np.save(r'data/my_test_img', images)

for path in (train_label_filepath, test_label_filepath):
    labels = []
    with open(path, 'rb') as file:
        magic, size = struct.unpack(">II", file.read(8))

        if magic != 2049:
            raise ValueError('Magic number mismatch, expected 2049, got {}'.format(magic))

        labels = np.frombuffer(file.read(), dtype=np.uint8)
        #labels = np.array(mapping(labels))


    if path == train_label_filepath:
        np.save(r'data/my_train_label', labels)
    else:
        np.save(r'data/my_test_label', labels)
