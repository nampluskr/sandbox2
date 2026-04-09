from .functions import softmax, cross_entropy, accuracy
from .functions import sigmoid, binary_cross_entropy, binary_accuracy
from .functions import identity, mse, r2_score


class MulticlassClassifier:
    def __init__(self, model, optimizer):
        self.model = model
        self.optimizer = optimizer

    def train_step(self, x, y):
        logits = self.model(x)
        preds = softmax(logits)
        loss = cross_entropy(preds, y)
        acc = accuracy(preds, y)

        dout = (preds - y) / x.shape[0]
        self.model.backward(dout)
        self.optimizer.step()
        return loss, acc

    def eval_step(self, x, y):
        logits = self.model(x)
        preds = softmax(logits)
        loss = cross_entropy(preds, y)
        acc = accuracy(preds, y)
        return loss, acc

    def predict(self, x):
        logits = self.model(x)
        preds = softmax(logits)
        return preds


class BinaryClassifier:
    def __init__(self, model, optimizer):
        self.model = model
        self.optimizer = optimizer

    def train_step(self, x, y):
        logits = self.model(x)
        preds = sigmoid(logits)
        loss = binary_cross_entropy(preds, y)
        acc = binary_accuracy(preds, y)

        dout = (preds - y) / x.shape[0]
        self.model.backward(dout)
        self.optimizer.step()
        return loss, acc

    def eval_step(self, x, y):
        logits = self.model(x)
        preds = sigmoid(logits)
        loss = binary_cross_entropy(preds, y)
        acc = binary_accuracy(preds, y)
        return loss, acc

    def predict(self, x):
        logits = self.model(x)
        preds = sigmoid(logits)
        return preds


class Regressor:
    def __init__(self, model, optimizer):
        self.model = model
        self.optimizer = optimizer

    def train_step(self, x, y):
        logits = self.model(x)
        preds = identity(logits)
        loss = mse(preds, y)
        acc = r2_score(preds, y)

        dout = 2 * (preds - y) / x.shape[0]
        self.model.backward(dout)
        self.optimizer.step()
        return loss, acc

    def eval_step(self, x, y):
        logits = self.model(x)
        preds = identity(logits)
        loss = mse(preds, y)
        acc = r2_score(preds, y)
        return loss, acc

    def predict(self, x):
        logits = self.model(x)
        preds = identity(logits)
        return preds


def train(model, dataloader):
    total_loss = 0
    total_acc = 0
    total_size = 0

    for x, y in dataloader:
        batch_size = len(x)
        total_size += batch_size

        loss, acc = model.train_step(x, y)
        total_loss += loss * batch_size
        total_acc += acc * batch_size
    return total_loss / total_size, total_acc / total_size


def evaluate(model, dataloader):
    total_loss = 0
    total_acc = 0
    total_size = 0

    for x, y in dataloader:
        batch_size = len(x)
        total_size += batch_size

        loss, acc = model.eval_step(x, y)
        total_loss += loss * batch_size
        total_acc += acc * batch_size
    return total_loss / total_size, total_acc / total_size


def predict(model, x):
    return model.predict(x)
