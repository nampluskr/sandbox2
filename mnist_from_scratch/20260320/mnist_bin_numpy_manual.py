import numpy as np

import common.mnist as mnist
from common.functions import one_hot, sigmoid, sigmoid_grad, softmax, cross_entropy, accuracy_fn
from common.functions import to_binary_label, binary_cross_entropy, binary_cross_entropy_grad, binary_accuracy


if __name__ == "__main__":

    #################################################################
    # Hyperparameters
    #################################################################
    DATA_DIR = "E:\\datasets\\mnist"
    SEED = 42
    BATCH_SIZE = 128
    LEARNING_RATE = 1e-2
    NUM_EPOCHS = 20
    NUM_SAMPLES = 10

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
    y_train = (y_train % 2).astype(np.float64).reshape(-1, 1)
    x_test = x_test.reshape(-1, 784).astype(np.float32)
    y_test = (y_test % 2).astype(np.float64).reshape(-1, 1)

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
        total_acc = 0
        total_size = 0

        indices = np.arange(len(x_train))
        np.random.shuffle(indices)

        for idx in range(0, len(x_train), BATCH_SIZE):
            x = x_train[indices[idx: idx + BATCH_SIZE]]
            y = y_train[indices[idx: idx + BATCH_SIZE]]

            batch_size = len(x)
            total_size += batch_size

            # Forward propagation
            z1 = np.dot(x, w1) + b1
            a1 = sigmoid(z1)
            z2 = np.dot(a1, w2) + b2
            a2 = sigmoid(z2)
            z3 = np.dot(a2, w3) + b3
            y_preds = sigmoid(z3)

            loss = binary_cross_entropy(y_preds, y)
            acc = binary_accuracy(y_preds, y)

            # Backward propagation
            dout = binary_cross_entropy_grad(y_preds, y)
            grad_z3 = sigmoid_grad(y_preds) * dout
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
              f"loss={total_loss/total_size:.3f} acc={total_acc/total_size:.3f}")

    #################################################################
    # Evaluation
    #################################################################
    print(f"\n>> Evaluation:")

    total_loss = 0
    total_acc = 0
    total_size = 0

    for idx in range(0, len(x_test), BATCH_SIZE):
        x = x_test[idx:idx + BATCH_SIZE]
        y = y_test[idx:idx + BATCH_SIZE]

        batch_size = len(x)
        total_size += batch_size

        # Forward propagation
        z1 = np.dot(x, w1) + b1
        a1 = sigmoid(z1)
        z2 = np.dot(a1, w2) + b2
        a2 = sigmoid(z2)
        z3 = np.dot(a2, w3) + b3
        y_preds = sigmoid(z3)

        loss = binary_cross_entropy(y_preds, y)
        acc = binary_accuracy(y_preds, y)

        total_loss += loss * batch_size
        total_acc += acc * batch_size

    print(f"loss={total_loss/total_size:.3f} acc={total_acc/total_size:.3f}")

    #################################################################
    # Prediction
    #################################################################
    print(f"\n>> Prediction:")
    label_str = {0: "even", 1: "odd"}

    x = x_test[:NUM_SAMPLES]
    z1 = np.dot(x, w1) + b1
    a1 = sigmoid(z1)
    z2 = np.dot(a1, w2) + b2
    a2 = sigmoid(z2)
    z3 = np.dot(a2, w3) + b3
    y_preds = sigmoid(z3)

    for i in range(NUM_SAMPLES):
        pred_label = int(y_preds[i, 0] >= 0.5)
        true_label = int(y[i, 0])
        print(f"Target: {true_label}({label_str[true_label]:<4}) | "
              f"Prediction: {pred_label}({label_str[pred_label]:<4}) "
              f"(prob_odd: {y_preds[i, 0]:.3f})")

    results = """

>> Training:
[ 1/10] loss:4.630 acc:0.175
[ 2/10] loss:2.577 acc:0.339
[ 3/10] loss:1.993 acc:0.443
[ 4/10] loss:1.681 acc:0.509
[ 5/10] loss:1.484 acc:0.557
[ 6/10] loss:1.344 acc:0.592
[ 7/10] loss:1.238 acc:0.621
[ 8/10] loss:1.154 acc:0.645
[ 9/10] loss:1.086 acc:0.663
[10/10] loss:1.029 acc:0.681

>> Evaluation:
[10/10] loss:1.000 acc:0.682

>> Prediction:
Target: 7 | Prediction: 7
Target: 2 | Prediction: 6
Target: 1 | Prediction: 1
Target: 0 | Prediction: 0
Target: 4 | Prediction: 4
Target: 1 | Prediction: 1
Target: 4 | Prediction: 4
Target: 9 | Prediction: 5
Target: 5 | Prediction: 6
Target: 9 | Prediction: 9
"""
