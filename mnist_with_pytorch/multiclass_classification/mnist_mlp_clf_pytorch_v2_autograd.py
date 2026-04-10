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

from common.mnist import load_images, load_labels, one_hot

#################################################################
# Functions
#################################################################
def sigmoid_grad(x):
    return x * (1 - x)

def accuracy(preds, targets):
    targets = targets.argmax(dim=1)
    return (preds.argmax(dim=1) == targets).float().mean()

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

w1 = torch.randn(784, 256, requires_grad=True)
b1 = torch.zeros(256, requires_grad=True)
w2 = torch.randn(256, 128, requires_grad=True)
b2 = torch.zeros(128, requires_grad=True)
w3 = torch.randn(128, 10, requires_grad=True)
b3 = torch.zeros(10, requires_grad=True)

#################################################################
# Training
#################################################################
print(f"\n>> Training:")

for epoch in range(1, NUM_EPOCHS + 1):
    total_loss = 0
    total_acc = 0
    total_size = 0

    indices = torch.randperm(len(x_train))

    for idx in range(0, len(x_train), BATCH_SIZE):
        x = x_train[indices[idx:idx + BATCH_SIZE]]  # (N, 784)
        y = y_train[indices[idx:idx + BATCH_SIZE]]  # (N, 10)
        batch_size = x.size(0)
        total_size += batch_size

        # Forward propagation
        z1 = torch.matmul(x, w1) + b1               # (N, 256)
        a1 = torch.sigmoid(z1)                      # (N, 256)
        z2 = torch.matmul(a1, w2) + b2              # (N, 128)
        a2 = torch.sigmoid(z2)                      # (N, 128)
        z3 = torch.matmul(a2, w3) + b3              # (N, 10)

        loss = F.cross_entropy(z3, y)               # (N, 10), (N, 10)
        preds = torch.softmax(z3, dim=1)            # (N, 10)
        acc = accuracy(preds, y)                    # (N, 10), (N, 10)

        # Backward propagation (autograd)
        loss.backward()

        # Update weights (in-place)
        with torch.no_grad():
            w1 -= LEARNING_RATE * w1.grad
            b1 -= LEARNING_RATE * b1.grad
            w2 -= LEARNING_RATE * w2.grad
            b2 -= LEARNING_RATE * b2.grad
            w3 -= LEARNING_RATE * w3.grad
            b3 -= LEARNING_RATE * b3.grad
            
            w1.grad.zero_()
            b1.grad.zero_()
            w2.grad.zero_()
            b2.grad.zero_()
            w3.grad.zero_()
            b3.grad.zero_()

        total_loss += loss.item() * batch_size
        total_acc += acc.item() * batch_size

    print(f"[{epoch:>2}/{NUM_EPOCHS}] "
          f"loss:{total_loss/total_size:.3f} acc:{total_acc/total_size:.3f}")

#################################################################
# Evaluaiton
#################################################################
print(f"\n>> Evaluation:")

total_loss = 0.0
total_acc = 0.0
total_size = 0

for idx in range(0, len(x_test), BATCH_SIZE):
    x = x_test[idx:idx + BATCH_SIZE]
    y = y_test[idx:idx + BATCH_SIZE]
    batch_size = x.size(0)
    total_size += batch_size

    # Forward propagation
    z1 = torch.matmul(x, w1) + b1
    a1 = torch.sigmoid(z1)
    z2 = torch.matmul(a1, w2) + b2
    a2 = torch.sigmoid(z2)
    z3 = torch.matmul(a2, w3) + b3

    loss = F.cross_entropy(z3, y)
    preds = torch.softmax(z3, dim=1)
    acc = accuracy(preds, y)

    total_loss += loss.item() * batch_size
    total_acc += acc.item() * batch_size

print(f"loss:{total_loss/total_size:.3f} acc:{total_acc/total_size:.3f}")

#################################################################
# Prediction
#################################################################
print(f"\n>> Prediction:")

x = x_test[:NUM_SAMPLES]
y = y_test[:NUM_SAMPLES]

z1 = torch.matmul(x, w1) + b1
a1 = torch.sigmoid(z1)
z2 = torch.matmul(a1, w2) + b2
a2 = torch.sigmoid(z2)
z3 = torch.matmul(a2, w3) + b3

preds = torch.softmax(z3, dim=1)

for i in range(NUM_SAMPLES):
    print(f"Target: {y[i].argmax()} | Prediction: {preds[i].argmax()}")
