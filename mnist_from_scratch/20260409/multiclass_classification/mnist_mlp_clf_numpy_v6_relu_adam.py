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
from common.functions import one_hot, sigmoid, sigmoid_grad
from common.functions import softmax, cross_entropy, accuracy
from common.modules import Linear, Sigmoid, Sequential, ReLU
from common.optimizer import SGD, Adam
from common.dataloader import Dataloader
from common.trainer import MulticlassClassifier, train, evaluate, predict

#################################################################
# Hyperparameters and preprocessing
#################################################################
DATA_DIR = r"E:\datasets\mnist"
SEED = 42
BATCH_SIZE = 64
LEARNING_RATE = 1e-3
NUM_EPOCHS = 10
NUM_SAMPLES = 10

#################################################################
# Data loading and preprocessing
#################################################################
x_train = load_images(DATA_DIR, "train")    # (60000, 28, 28)
y_train = load_labels(DATA_DIR, "train")    # (60000,)
x_test = load_images(DATA_DIR, "test")      # (10000, 28, 28)
y_test = load_labels(DATA_DIR, "test")      # (10000,)

x_train = x_train.reshape(-1, 784).astype(np.float32) / 255.0   # (60000, 784)
x_test = x_test.reshape(-1, 784).astype(np.float32) / 255.0     # (10000, 784)

y_train = one_hot(y_train, num_classes=10).astype(np.float32)   # (60000, 10)
y_test = one_hot(y_test, num_classes=10).astype(np.float32)     # (10000, 10)

#################################################################
# Modeling and training
#################################################################
np.random.seed(SEED)

train_loader = Dataloader(x_train, y_train, batch_size=BATCH_SIZE, shuffle=True)
test_loader = Dataloader(x_test, y_test, batch_size=BATCH_SIZE, shuffle=False)

model = Sequential(
    Linear(784, 256),
    ReLU(),
    Linear(256, 128),
    ReLU(),
    Linear(128, 10),
)
model.layers[0].w *= np.sqrt(2 / 784)   # He(Kaiming) initialization (for ReLU, LeakyReLU)
model.layers[2].w *= np.sqrt(2 / 256)   # He(Kaiming) initialization (for ReLU, LeakyReLU)
model.layers[4].w *= np.sqrt(2 / 128)   # He(Kaiming) initialization (for ReLU, LeakyReLU)

optimizer = Adam(model, lr=0.001)
clf = MulticlassClassifier(model, optimizer)

print(f"\n>> Training:")
for epoch in range(1, NUM_EPOCHS + 1):
    loss, acc = train(clf, train_loader)
    print(f"[{epoch:>2}/{NUM_EPOCHS}] loss:{loss:.3f} acc:{acc:.3f}")

#################################################################
# Evaluation
#################################################################
print(f"\n>> Evaluation:")

loss, acc = evaluate(clf, test_loader)
print(f"loss:{loss:.3f} acc:{acc:.3f}")

#################################################################
# Prediction
#################################################################
print(f"\n>> Prediction:")

x = x_test[:NUM_SAMPLES]
y = y_test[:NUM_SAMPLES]

preds = predict(clf, x)

for i in range(NUM_SAMPLES):
    print(f"Target: {y[i].argmax()} | Prediction: {preds[i].argmax()}")
