import sys
import os

FILE_DIR = os.path.dirname(os.path.abspath(__file__))
CONFIG_DIR = os.path.normpath(os.path.join(FILE_DIR, "..", "..", "configs"))
SOURCE_DIR = os.path.normpath(os.path.join(FILE_DIR, "..", "..", "src"))

if SOURCE_DIR not in sys.path:
    sys.path.insert(0, SOURCE_DIR)
    
#################################################################
## 1. Hyperparameters
#################################################################
import numpy as np
from dotenv import load_dotenv
import yaml

from common.mnist import load_images, load_labels, one_hot
from common.functions import sigmoid, sigmoid_grad
from common.functions import softmax, cross_entropy, accuracy

load_dotenv()
with open(os.path.join(CONFIG_DIR, "multiclass.yaml"), "r") as f:
    config = yaml.safe_load(f)

DATA_DIR = os.environ["DATASET_DIR"]
SEED = config["SEED"]
BATCH_SIZE = config["BATCH_SIZE"]
LEARNING_RATE = float(config["LEARNING_RATE"])
NUM_EPOCHS = config["NUM_EPOCHS"]
NUM_SAMPLES = config["NUM_SAMPLES"]

#################################################################
## 2. Data loading and preprocessing
#################################################################
x_train = load_images(DATA_DIR, "train")                        # (60000, 28, 28)
y_train = load_labels(DATA_DIR, "train")                        # (60000,)
x_test = load_images(DATA_DIR, "test")                          # (10000, 28, 28)
y_test = load_labels(DATA_DIR, "test")                          # (10000,)

x_train = x_train.reshape(-1, 784).astype(np.float32) / 255.0   # (60000, 784)
x_test = x_test.reshape(-1, 784).astype(np.float32) / 255.0     # (10000, 784)

y_train = one_hot(y_train, num_classes=10).astype(np.float32)   # (60000, 10)
y_test = one_hot(y_test, num_classes=10).astype(np.float32)     # (10000, 10)

#################################################################
## 3. Modeling
#################################################################
np.random.seed(SEED)

w1 = np.random.randn(784, 256)
b1 = np.zeros(256)
w2 = np.random.randn(256, 128)
b2 = np.zeros(128)
w3 = np.random.randn(128, 10)
b3 = np.zeros(10)

#################################################################
## 4. Training
#################################################################
print(f"\n>> Training:")

for epoch in range(1, NUM_EPOCHS + 1):
    total_loss = 0
    total_acc = 0
    total_size = 0

    indices = np.random.permutation(len(x_train))

    for idx in range(0, len(x_train), BATCH_SIZE):
        x = x_train[indices[idx: idx + BATCH_SIZE]]
        y = y_train[indices[idx: idx + BATCH_SIZE]]
        batch_size = len(x)
        total_size += batch_size

        # Forward propagation (multiclass: softmax + cross_entropy)
        z1 = np.dot(x, w1) + b1                     # (N, 256)
        a1 = sigmoid(z1)                            # (N, 256)
        z2 = np.dot(a1, w2) + b2                    # (N, 128)
        a2 = sigmoid(z2)                            # (N, 128)
        z3 = np.dot(a2, w3) + b3                    # (N, 10)

        preds = softmax(z3)                         # (N, 10)
        loss = cross_entropy(preds, y)              # (N, 10), (N, 10)
        acc = accuracy(preds, y)                    # (N, 10), (N, 10)

        # Backward propagation (manual)
        # dout = cross_entory_grad(y_preds, y)      # (N, 10)
        # grad_z3 = softmax_grad(y_preds) * dout    # (N, 10)
        grad_z3 = (preds - y) / batch_size          # (N, 10)
        grad_w3 = np.dot(a2.T, grad_z3)             # (128, 10)
        grad_b3 = np.sum(grad_z3, axis=0)           # (10, )

        grad_a2 = np.dot(grad_z3, w3.T)             # (N, 128)
        grad_z2 = sigmoid_grad(a2) * grad_a2        # (N, 128)
        grad_w2 = np.dot(a1.T, grad_z2)             # (256, 128)
        grad_b2 = np.sum(grad_z2, axis=0)           # (128,)

        grad_a1 = np.dot(grad_z2, w2.T)             # (N, 256)
        grad_z1 = sigmoid_grad(a1) * grad_a1        # (N, 256)
        grad_w1 = np.dot(x.T, grad_z1)              # (784, 256)
        grad_b1 = np.sum(grad_z1, axis=0)           # (256,)

        # Update weights
        w1 -= LEARNING_RATE * grad_w1
        b1 -= LEARNING_RATE * grad_b1
        w2 -= LEARNING_RATE * grad_w2
        b2 -= LEARNING_RATE * grad_b2
        w3 -= LEARNING_RATE * grad_w3
        b3 -= LEARNING_RATE * grad_b3

        total_loss += loss * batch_size
        total_acc += acc * batch_size

    print(f"[{epoch:>2}/{NUM_EPOCHS}] "
          f"loss:{total_loss/total_size:.3f} acc:{total_acc/total_size:.3f}")
    
#################################################################
## 5. Evaluaton
#################################################################
print(f"\n>> Evaluation:")

total_loss = 0.0
total_acc = 0.0
total_size = 0

for idx in range(0, len(x_test), BATCH_SIZE):
    x = x_test[idx:idx + BATCH_SIZE]
    y = y_test[idx:idx + BATCH_SIZE]
    batch_size = x.shape[0]
    total_size += batch_size

    z1 = np.dot(x, w1) + b1
    a1 = sigmoid(z1)
    z2 = np.dot(a1, w2) + b2
    a2 = sigmoid(z2)
    z3 = np.dot(a2, w3) + b3

    preds = softmax(z3)
    loss = cross_entropy(preds, y)
    acc = accuracy(preds, y)

    total_loss += loss * batch_size
    total_acc += acc * batch_size

print(f"loss:{total_loss/total_size:.3f} acc:{total_acc/total_size:.3f}")

#################################################################
## 6. Prediction
#################################################################
print(f"\n>> Prediction:")

x = x_test[:NUM_SAMPLES]
y = y_test[:NUM_SAMPLES]

z1 = np.dot(x, w1) + b1
a1 = sigmoid(z1)
z2 = np.dot(a1, w2) + b2
a2 = sigmoid(z2)
z3 = np.dot(a2, w3) + b3

preds = softmax(z3)

for i in range(NUM_SAMPLES):
    print(f"Target: {y[i].argmax()} | Prediction: {preds[i].argmax()}")
