import numpy as np
from .functions import sigmoid, im2col, col2im


class Module:
    def __init__(self):
        self.params = []
        self.grads = []
        self.training = True

    def __call__(self, *args):
        return self.forward(*args)

    def forward(self, *args):
        raise NotImplementedError

    def train(self):
        self.training = True

    def eval(self):
        self.training = False


class Linear(Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.w = np.random.randn(in_features, out_features)
        self.b = np.zeros(out_features)
        self.grad_w = np.zeros_like(self.w)
        self.grad_b = np.zeros_like(self.b)

        self.params.extend([self.w, self.b])
        self.grads.extend([self.grad_w, self.grad_b])
        self.x = None

    def forward(self, x):
        self.x = x
        return np.dot(x, self.w) + self.b

    def backward(self, dout):
        self.grad_w[...] = np.dot(self.x.T, dout)
        self.grad_b[...] = np.sum(dout, axis=0)
        return np.dot(dout, self.w.T)


class Sigmoid(Module):
    def forward(self, x):
        self.out = sigmoid(x)
        return self.out

    def backward(self, dout):
        return dout * self.out * (1 - self.out)


class ReLU(Module):
    def forward(self, x):
        self.mask = x > 0
        return x * self.mask

    def backward(self, dout):
        return dout * self.mask


class Sequential(Module):
    def __init__(self, *layers):
        super().__init__()
        self.layers = list(layers)

        for layer in self.layers:
            self.params.extend(layer.params)
            self.grads.extend(layer.grads)

    def forward(self, x):
        for layer in self.layers:
            x = layer.forward(x)
        return x

    def backward(self, dout):
        for layer in reversed(self.layers):
            dout = layer.backward(dout)

    def train(self):
        self.training = True
        for layer in self.layers:
            layer.train()

    def eval(self):
        self.training = False
        for layer in self.layers:
            layer.eval()


class Conv2d(Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding

        scale = np.sqrt(1. / (in_channels * kernel_size * kernel_size))
        self.w = np.random.randn(out_channels, in_channels, kernel_size, kernel_size) * scale
        self.b = np.zeros(out_channels)
        self.grad_w = np.zeros_like(self.w)
        self.grad_b = np.zeros_like(self.b)

        self.params = [self.w, self.b]
        self.grads = [self.grad_w, self.grad_b]

        self.x = None
        self.col_cache = None  # (col_x, out_h, out_w)
        self.col_w = None

    def forward(self, x):
        B, C, H, W = x.shape
        self.x = x

        col_x, out_h, out_w = im2col(x, self.kernel_size, self.stride, self.padding)
        self.col_cache = (col_x, out_h, out_w)
        self.col_w = self.w.reshape(self.out_channels, -1)  # (out_c, in_c * K * K)

        out = np.dot(col_x, self.col_w.T) + self.b          # (B*out_h*out_w, out_c)
        out = out.reshape(B, out_h, out_w, self.out_channels).transpose(0, 3, 1, 2)
        return out

    def backward(self, dout):
        B, out_c, out_h, out_w = dout.shape
        col_x, out_h_cache, out_w_cache = self.col_cache
        dout_flat = dout.transpose(0, 2, 3, 1).reshape(-1, self.out_channels)
        self.grad_b[...] = np.sum(dout_flat, axis=0)
        grad_w_flat = np.dot(dout_flat.T, col_x)
        self.grad_w[...] = grad_w_flat.reshape(self.grad_w.shape)
        col_w = self.w.reshape(self.out_channels, -1)  # 재계산
        dcol_x = np.dot(dout_flat, col_w)
        dx = col2im(dcol_x, self.x.shape, self.kernel_size, self.stride, self.padding)
        return dx


class Dropout(Module):
    def __init__(self, p=0.5):
        super().__init__()
        self.p = p
        self.mask = None

    def forward(self, x):
        if self.training:
            self.mask = (np.random.rand(*x.shape) > self.p) / (1.0 - self.p)
            return x * self.mask
        return x

    def backward(self, dout):
        if self.training:
            return dout * self.mask
        return dout


class MaxPool2d(Module):
    def __init__(self, kernel_size, stride=None, padding=0):
        super().__init__()
        self.kernel_size = kernel_size
        self.stride = stride if stride is not None else kernel_size
        self.padding = padding
        self.cache = None

    def forward(self, x):
        B, C, H, W = x.shape
        col_x, out_h, out_w = im2col(x, self.kernel_size, self.stride, self.padding)
        col_x = col_x.reshape(-1, self.kernel_size * self.kernel_size)

        self.out_h, self.out_w = out_h, out_w
        self.input_shape = x.shape
        self.max_indices = np.argmax(col_x, axis=1)
        output = np.max(col_x, axis=1).reshape(B, out_h, out_w, C).transpose(0, 3, 1, 2)

        self.cache = col_x
        return output

    def backward(self, dout):
        B, C, out_h, out_w = dout.shape
        dout_flat = dout.transpose(0, 2, 3, 1).flatten()

        dcol = np.zeros_like(self.cache)
        dcol[np.arange(self.max_indices.size), self.max_indices] = dout_flat

        dx = col2im(dcol, self.input_shape, self.kernel_size, self.stride, self.padding)
        return dx


class Flatten(Module):
    def forward(self, x):
        self.input_shape = x.shape
        return x.reshape(x.shape[0], -1)

    def backward(self, dout):
        return dout.reshape(self.input_shape)
