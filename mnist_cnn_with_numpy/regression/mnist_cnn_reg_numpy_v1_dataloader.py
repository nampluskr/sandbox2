import os
import sys

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

#################################################################
# Import libraries
#################################################################
import numpy as np

from common.mnist import load_images, load_labels
from common.functions import sigmoid, sigmoid_grad
from common.functions import identity, mse, r2_score
from common.modules import Linear, Sigmoid, Sequential
from common.modules import Conv2d, ReLU, MaxPool2d, Dropout, Flatten
from common.optimizer import SGD, Adam
from common.dataloader import Dataloader

#################################################################
# Hyperparameters and preprocessing
#################################################################
DATA_DIR = r"E:\datasets\mnist"
SEED = 42
BATCH_SIZE = 32
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

x_train = x_train.reshape(-1, 1, 28, 28).astype(np.float32) / 255.0   # (60000, 1, 28, 28)
x_test = x_test.reshape(-1, 1, 28, 28).astype(np.float32) / 255.0     # (10000, 1, 28, 28)

y_train = y_train.astype(np.float32).reshape(-1, 1) / 9.0       # (60000, 1)
y_test = y_test.astype(np.float32).reshape(-1, 1) / 9.0         # (10000, 1)

#################################################################
# Modeling and training
#################################################################
np.random.seed(SEED)

train_loader = Dataloader(x_train, y_train, batch_size=BATCH_SIZE, shuffle=True)
test_loader = Dataloader(x_test, y_test, batch_size=BATCH_SIZE, shuffle=False)

model = Sequential(
    Conv2d(1, 16, kernel_size=3, stride=1, padding=1),  # (N, 16, 28, 28)
    ReLU(),
    MaxPool2d(kernel_size=2, stride=2),                 # (N, 16, 14, 14)
    Conv2d(16, 32, kernel_size=3, stride=1, padding=1), # (N, 32, 14, 14)
    ReLU(),
    MaxPool2d(kernel_size=2, stride=2),                 # (N, 32, 7, 7)
    Flatten(),
    Dropout(p=0.5),
    Linear(32 * 7 * 7, 1)
)
optimizer = Adam(model, lr=LEARNING_RATE)

print(f"\n>> Training:")
for epoch in range(1, NUM_EPOCHS + 1):
    total_loss = 0
    total_acc = 0
    total_size = 0

    for x, y in train_loader:
        batch_size = x.shape[0]
        total_size += batch_size

        # Forward propagation (regression: identity + mse)
        logits = model(x)                           # (N, 1)
        preds = identity(logits)                    # (N, 1)
        loss = mse(preds, y)                        # (N, 1), (N, 1)
        acc = r2_score(preds, y)                    # (N, 1), (N, 1)

        # Backward propagation (manual)
        dout = 2 * (preds - y) / batch_size         # (N, 10)
        model.backward(dout)

        # Update weights (in-place)
        optimizer.step()

        total_loss += loss * batch_size
        total_acc += acc * batch_size

    print(f"[{epoch:>2}/{NUM_EPOCHS}] "
          f"loss:{total_loss/total_size:.3f} acc:{total_acc/total_size:.3f}")

#################################################################
# Evaluaiton
#################################################################
print(f"\n>> Evaluation:")

total_loss = 0.0
total_acc = 0.0
total_size = 0

for x, y in test_loader:
    batch_size = x.shape[0]
    total_size += batch_size

    logits = model(x)
    preds = identity(logits)
    loss = mse(preds, y)
    acc = r2_score(preds, y)

    total_loss += loss * batch_size
    total_acc += acc * batch_size

print(f"loss:{total_loss/total_size:.3f} acc:{total_acc/total_size:.3f}")

#################################################################
# Prediction
#################################################################
print(f"\n>> Prediction:")

x = x_test[:NUM_SAMPLES]
y = y_test[:NUM_SAMPLES]

logits = model(x)
preds = identity(logits)

for i in range(NUM_SAMPLES):
    raw = preds[i, 0]
    pred_label = int(np.round(np.clip(raw * 9.0, 0, 9)))
    true_label = int(np.round(y[i, 0] * 9.0))
    print(f"Target: {true_label} | Prediction: {pred_label} (raw: {raw:.4f})")
