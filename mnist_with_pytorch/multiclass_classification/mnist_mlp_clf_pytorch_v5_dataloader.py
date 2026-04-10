import os
import sys

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

#################################################################
# Import libraries
#################################################################
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from common.mnist import load_images, load_labels, one_hot

#################################################################
# Functions
#################################################################
def sigmoid_grad(x):
    return x * (1 - x)

def accuracy(preds, targets):
    targets = targets.argmax(dim=1)
    return (preds.argmax(dim=1) == targets).float().mean()

class ImageDataset(Dataset):
    def __init__(self, images, labels):
        self.images = images
        self.labels = labels
        
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        image = self.images[idx]
        label = self.labels[idx]
        return image, label

#################################################################
# Hyperparameters and preprocessing
#################################################################
DATA_DIR = r"E:\datasets\mnist"
SEED = 42
BATCH_SIZE = 64
LEARNING_RATE = 1e-2
NUM_EPOCHS = 10
NUM_SAMPLES = 10

#################################################################
# Data loading and preprocessing
#################################################################
x_train = load_images(DATA_DIR, "train")    # (60000, 28, 28)
y_train = load_labels(DATA_DIR, "train")    # (60000,)
x_test = load_images(DATA_DIR, "test")      # (10000, 28, 28)
y_test = load_labels(DATA_DIR, "test")      # (10000,)

x_train_np = x_train.reshape(-1, 784).astype(np.float32) / 255.0
x_test_np = x_test.reshape(-1, 784).astype(np.float32) / 255.0
y_train_np = one_hot(y_train, num_classes=10).astype(np.float32)
y_test_np = one_hot(y_test, num_classes=10).astype(np.float32)

x_train = torch.from_numpy(x_train_np)          # (60000, 784)
y_train = torch.from_numpy(y_train_np)          # (60000, 10)
x_test = torch.from_numpy(x_test_np)            # (10000, 784)
y_test = torch.from_numpy(y_test_np)            # (10000, 10)

#################################################################
# Modeling
#################################################################
torch.manual_seed(SEED)

class Linear(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.w = nn.Parameter(torch.randn(in_features, out_features))
        self.b = nn.Parameter(torch.zeros(out_features))

    def forward(self, x):
        return torch.matmul(x, self.w) + self.b

# model = nn.Sequential(
#     Linear(784, 256),
#     nn.Sigmoid(),
#     Linear(256, 128),
#     nn.Sigmoid(),
#     Linear(128, 10),
# )

class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear1 = Linear(784, 256)
        self.linear2 = Linear(256, 128)
        self.linear3 = Linear(128, 10)

    def forward(self, x):
        x = torch.sigmoid(self.linear1(x))
        x = torch.sigmoid(self.linear2(x))
        x = self.linear3(x)
        return x

model = MLP()
optimizer = torch.optim.SGD(model.parameters(), lr=LEARNING_RATE)

train_loader = DataLoader(ImageDataset(x_train, y_train), batch_size=BATCH_SIZE, shuffle=True)
test_loader = DataLoader(ImageDataset(x_test, y_test), batch_size=BATCH_SIZE, shuffle=False)

#################################################################
# Training
#################################################################
print(f"\n>> Training:")

for epoch in range(1, NUM_EPOCHS + 1):
    total_loss = 0
    total_acc = 0
    total_size = 0

    for x, y in train_loader:
        batch_size = x.size(0)
        total_size += batch_size

        logits = model(x)
        loss = F.cross_entropy(logits, y)           # (N, 10), (N, 10)
        preds = torch.softmax(logits, dim=1)        # (N, 10)
        acc = accuracy(preds, y)                    # (N, 10), (N, 10)

        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        total_loss += loss.item() * batch_size
        total_acc += acc.item() * batch_size

    print(f"[{epoch:>2}/{NUM_EPOCHS}] "
          f"loss:{total_loss/total_size:.3f} acc:{total_acc/total_size:.3f}")

#################################################################
# Evaluaiton
#################################################################
print(f"\n>> Evaluation:")

model.eval()
with torch.no_grad():
    total_loss = 0.0
    total_acc = 0.0
    total_size = 0

    for x, y in test_loader:
        batch_size = x.size(0)
        total_size += batch_size

        logits = model(x)
        loss = F.cross_entropy(logits, y)
        preds = torch.softmax(logits, dim=1)
        acc = accuracy(preds, y)

        total_loss += loss.item() * batch_size
        total_acc += acc.item() * batch_size

print(f"loss:{total_loss/total_size:.3f} acc:{total_acc/total_size:.3f}")

#################################################################
# Prediction
#################################################################
print(f"\n>> Prediction:")

model.eval()
with torch.no_grad():
    x = x_test[:NUM_SAMPLES]
    y = y_test[:NUM_SAMPLES]

    logits = model(x)
    preds = torch.softmax(logits, dim=1)

    for i in range(NUM_SAMPLES):
        print(f"Target: {y[i].argmax()} | Prediction: {preds[i].argmax()}")
