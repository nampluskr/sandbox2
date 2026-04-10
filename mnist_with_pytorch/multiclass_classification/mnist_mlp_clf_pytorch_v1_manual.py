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

from common.mnist import load_images, load_labels, one_hot

#################################################################
# Functions
#################################################################
def sigmoid_grad(x):
    return x * (1 - x)

def softmax(x):
    # x: (N, num_classes)
    x_max = torch.max(x, dim=1, keepdims=True).values
    e_x = torch.exp(x - x_max)
    return e_x / torch.sum(e_x, dim=1, keepdims=True)

def cross_entropy(preds, targets):
    # preds/targets: (N, num_classes)
    probs = torch.sum(preds * targets, dim=1)
    return -torch.mean(torch.log(probs))

def accuracy(preds, targets):
    # preds/targets: (N, num_classes)
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

w1 = torch.randn(784, 256)
b1 = torch.zeros(256)
w2 = torch.randn(256, 128)
b2 = torch.zeros(128)
w3 = torch.randn(128, 10)
b3 = torch.zeros(10)

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

        preds = softmax(z3)                         # (N, 10)
        loss = cross_entropy(preds, y)              # (N, 10), (N, 10)
        acc = accuracy(preds, y)                    # (N, 10), (N, 10)

        # Backward propagation (manual)
        # dout = cross_entory_grad(y_preds, y)      # (N, 10)
        # grad_z3 = softmax_grad(y_preds) * dout    # (N, 10)
        grad_z3 = (preds - y) / batch_size          # (N, 10)
        grad_w3 = torch.matmul(a2.t(), grad_z3)     # (128, 10)
        grad_b3 = torch.sum(grad_z3, dim=0)         # (10,)

        grad_a2 = torch.matmul(grad_z3, w3.t())     # (N, 128)
        grad_z2 = sigmoid_grad(a2) * grad_a2        # (N, 128)
        grad_w2 = torch.matmul(a1.t(), grad_z2)     # (256, 128)
        grad_b2 = torch.sum(grad_z2, dim=0)         # (128,)

        grad_a1 = torch.matmul(grad_z2, w2.t())     # (N, 256)
        grad_z1 = sigmoid_grad(a1) * grad_a1        # (N, 256)
        grad_w1 = torch.matmul(x.t(), grad_z1)      # (784, 256)
        grad_b1 = torch.sum(grad_z1, dim=0)         # (256,)

        # Update weights (in-place)
        w1 -= LEARNING_RATE * grad_w1
        b1 -= LEARNING_RATE * grad_b1
        w2 -= LEARNING_RATE * grad_w2
        b2 -= LEARNING_RATE * grad_b2
        w3 -= LEARNING_RATE * grad_w3
        b3 -= LEARNING_RATE * grad_b3

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

    preds = softmax(z3)
    loss = cross_entropy(preds, y)
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

preds = softmax(z3)

for i in range(NUM_SAMPLES):
    print(f"Target: {y[i].argmax()} | Prediction: {preds[i].argmax()}")
