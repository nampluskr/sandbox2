import os
import gzip
import numpy as np


def get_class_names(mnist_type="mnist"):
    return ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']


def load_images(data_dir, split="train"):
    filename = "train-images-idx3-ubyte.gz" if split == "train" else "t10k-images-idx3-ubyte.gz"
    filepath = os.path.join(data_dir, filename)
    with gzip.open(filepath, 'rb') as f:
        data = np.frombuffer(f.read(), np.uint8, offset=16)
    return data.reshape(-1, 28, 28).copy()


def load_labels(data_dir, split="train"):
    filename = "train-labels-idx1-ubyte.gz" if split == "train" else "t10k-labels-idx1-ubyte.gz"
    filepath = os.path.join(data_dir, filename)
    with gzip.open(filepath, 'rb') as f:
        data = np.frombuffer(f.read(), np.uint8, offset=8)
    return data


class Dataloader:
    def __init__(self, images, labels, batch_size, shuffle=False, drop_last=False):
        self.images = images
        self.labels = labels
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.drop_last = drop_last
        self.num_images = len(self.images)

        if drop_last:
            self.num_batches = self.num_images // batch_size
        else:
            self.num_batches = (self.num_images + batch_size - 1) // batch_size

    def __len__(self):
        return self.num_batches

    def __iter__(self):
        indices = np.arange(self.num_images)
        if self.shuffle:
            np.random.shuffle(indices)
        if self.drop_last:
            indices = indices[:self.num_batches * self.batch_size]

        for i in range(self.num_batches):
            idx = indices[i * self.batch_size:(i + 1) * self.batch_size]
            yield self.images[idx], self.labels[idx]
