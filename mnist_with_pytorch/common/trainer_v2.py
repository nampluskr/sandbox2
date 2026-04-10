import torch
import torch.nn as nn
import torch.nn.functional as F


def accuracy(logits, targets):
    preds = torch.softmax(logits, dim=1)
    targets = targets.argmax(dim=1)
    return (preds.argmax(dim=1) == targets).float().mean()


class BaseWrapper:
    def __init__(self, model, optimizer, device=None):
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = model.to(self.device)
        self.optimizer = optimizer
        self.loss_fn = None
        self.metric_fn = None
    
    def train(self):
        self.model.train()
        
    def eval(self):
        self.model.eval()    

    def train_step(self, x, y):
        self.train()
        x, y = x.to(self.device), y.to(self.device)
        logits = self.model(x)
        loss = self.loss_fn(logits, y)
        acc = self.metric_fn(logits, y)

        loss.backward()
        self.optimizer.step()
        self.optimizer.zero_grad()
        return loss.item(), acc.item()

    @torch.no_grad()
    def eval_step(self, x, y):
        self.eval()
        x, y = x.to(self.device), y.to(self.device)
        logits = self.model(x)
        loss = self.loss_fn(logits, y)
        acc = self.metric_fn(logits, y)
        return loss.item(), acc.item()
    
    @torch.no_grad()
    def predict(self, x):
        self.eval()
        x = x.to(self.device)
        logits = self.model(x)
        return logits.cpu()


class MulticlassClassifier(BaseWrapper):
    def __init__(self, model, optimizer, device=None):
        super().__init__(model, optimizer, device)
        self.loss_fn = F.cross_entropy
        self.metric_fn = accuracy

    @torch.no_grad()
    def predict(self, x):
        logits = super().predict(x)
        return torch.softmax(logits, dim=1)


def train(model, dataloader):
    model.train()
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
    model.eval()
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
    model.eval()
    return model.predict(x)
