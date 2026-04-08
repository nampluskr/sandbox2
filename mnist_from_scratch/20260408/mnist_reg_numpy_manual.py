import numpy as np

import common.mnist as mnist
from common.functions import sigmoid, sigmoid_grad
from common.functions import identity, identity_grad, mse, mse_grad, r2_score


if __name__ == "__main__":

    #################################################################
    # Hyperparameters
    #################################################################
    DATA_DIR = "E:\\datasets\\mnist"
    SEED = 42
    BATCH_SIZE = 128
    LEARNING_RATE = 1e-1
    NUM_EPOCHS = 100
    NUM_SAMPLES = 20

    np.random.seed(SEED)

    #################################################################
    # Data loading
    #################################################################
    x_train = mnist.load_images(DATA_DIR, "train")
    y_train = mnist.load_labels(DATA_DIR, "train")
    x_test = mnist.load_images(DATA_DIR, "test")
    y_test = mnist.load_labels(DATA_DIR, "test")

    #################################################################
    # Data Preprocessing
    #################################################################
    x_train = x_train.reshape(-1, 784).astype(np.float32) / 255.0
    x_test = x_test.reshape(-1, 784).astype(np.float32) / 255.0

    y_train = y_train.astype(np.float32).reshape(-1, 1) / 9.0
    y_test = y_test.astype(np.float32).reshape(-1, 1) / 9.0

    #################################################################
    # Modeling
    #################################################################
    w1 = np.random.randn(784, 256) * np.sqrt(2 / 784)
    b1 = np.zeros(256)
    w2 = np.random.randn(256, 128) * np.sqrt(2 / 256)
    b2 = np.zeros(128)
    w3 = np.random.randn(128, 1) * np.sqrt(2 / 128)
    b3 = np.zeros(1)

    #################################################################
    # Training
    #################################################################
    print(f"\n>> Training:")

    for epoch in range(1, NUM_EPOCHS + 1):
        total_loss = 0
        total_acc  = 0
        total_size = 0

        indices = np.arange(len(x_train))
        np.random.shuffle(indices)

        for idx in range(0, len(x_train), BATCH_SIZE):
            x = x_train[indices[idx: idx + BATCH_SIZE]]
            y = y_train[indices[idx: idx + BATCH_SIZE]]

            batch_size  = len(x)
            total_size += batch_size

            # Forward propagation
            z1 = np.dot(x, w1) + b1
            a1 = sigmoid(z1)
            z2 = np.dot(a1, w2) + b2
            a2 = sigmoid(z2)
            z3 = np.dot(a2, w3) + b3
            y_preds = identity(z3)

            loss = mse(y_preds, y)
            acc  = r2_score(y_preds, y)

            # Backward propagation (regression: identity + mse)
            # dout = mse_grad(y_preds, y)
            # grad_z3 = identity_grad(y_preds) * dout
            grad_z3 = 2 * (y_preds - y) / batch_size
            grad_w3 = np.dot(a2.T, grad_z3)
            grad_b3 = np.sum(grad_z3, axis=0)

            grad_a2 = np.dot(grad_z3, w3.T)
            grad_z2 = sigmoid_grad(a2) * grad_a2
            grad_w2 = np.dot(a1.T, grad_z2)
            grad_b2 = np.sum(grad_z2, axis=0)

            grad_a1 = np.dot(grad_z2, w2.T)
            grad_z1 = sigmoid_grad(a1) * grad_a1
            grad_w1 = np.dot(x.T, grad_z1)
            grad_b1 = np.sum(grad_z1, axis=0)

            # Update weights and biases
            w1 -= LEARNING_RATE * grad_w1
            b1 -= LEARNING_RATE * grad_b1
            w2 -= LEARNING_RATE * grad_w2
            b2 -= LEARNING_RATE * grad_b2
            w3 -= LEARNING_RATE * grad_w3
            b3 -= LEARNING_RATE * grad_b3

            total_loss += loss * batch_size
            total_acc += acc * batch_size

        print(f"[{epoch:>2}/{NUM_EPOCHS}] "
              f"loss={total_loss/total_size:.4f} acc={total_acc/total_size:.3f}")

    #################################################################
    # Evaluation
    #################################################################
    print(f"\n>> Evaluation:")

    total_loss = 0
    total_acc  = 0
    total_size = 0

    for idx in range(0, len(x_test), BATCH_SIZE):
        x = x_test[idx: idx + BATCH_SIZE]
        y = y_test[idx: idx + BATCH_SIZE]

        batch_size  = len(x)
        total_size += batch_size

        # Forward propagation
        z1 = np.dot(x, w1) + b1
        a1 = sigmoid(z1)
        z2 = np.dot(a1, w2) + b2
        a2 = sigmoid(z2)
        z3 = np.dot(a2, w3) + b3
        y_preds = identity(z3)

        loss = mse(y_preds, y)
        acc = r2_score(y_preds, y)

        total_loss += loss * batch_size
        total_acc += acc * batch_size

    print(f"loss={total_loss/total_size:.4f} acc={total_acc/total_size:.3f}")

    #################################################################
    # Prediction
    #################################################################
    print(f"\n>> Prediction:")

    x = x_test[:NUM_SAMPLES]
    y = y_test[:NUM_SAMPLES]

    z1 = np.dot(x, w1) + b1
    a1 = sigmoid(z1)
    z2 = np.dot(a1, w2) + b2
    a2 = sigmoid(z2)
    z3 = np.dot(a2, w3) + b3
    y_preds = identity(z3)

    for i in range(NUM_SAMPLES):
        raw = y_preds[i, 0]
        pred_label = int(np.round(np.clip(raw * 9.0, 0, 9)))
        true_label = int(np.round(y[i, 0] * 9.0))
        print(f"Target: {true_label} | Prediction: {pred_label} (raw: {raw:.4f})")
