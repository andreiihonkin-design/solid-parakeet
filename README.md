
```python
import numpy as np

class SimpleNN:
    def __init__(self, input_size, hidden_size, output_size, lr=0.01):
        self.w1 = np.random.randn(input_size, hidden_size) * 0.1
        self.w2 = np.random.randn(hidden_size, output_size) * 0.1
        self.lr = lr

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def forward(self, x):
        self.z1 = x @ self.w1
        self.a1 = self.sigmoid(self.z1)
        self.z2 = self.a1 @ self.w2
        self.a2 = self.sigmoid(self.z2)
        return self.a2

    def backward(self, x, y, output):
        error = output - y
        d_w2 = self.a1.T @ error
        d_hidden = error @ self.w2.T * (self.a1 * (1 - self.a1))
        d_w1 = x.T @ d_hidden

        self.w1 -= self.lr * d_w1
        self.w2 -= self.lr * d_w2

    def train(self, x, y, epochs=1000):
        for i in range(epochs):
            output = self.forward(x)
            self.backward(x, y, output)