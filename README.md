# Numpy Neural Network (Works on iOS Juno)

Simple fully-trainable neural network written in pure NumPy.
Works on iPhone using the Juno app (no PyTorch required).

## Features
- Pure NumPy neural network
- Trains on MNIST
- Runs offline on iOS
- Lightweight model (tiny MLP)
- Saving and loading weights

## Run training:

```python
from nn_numpy import load_mnist, TinyNN

x_train, y_train, x_test, y_test = load_mnist()

net = TinyNN()
net.train(x_train, y_train, epochs=3)

preds = net.predict(x_test)
acc = (preds == y_test).mean()
print("Accuracy:", acc)
```

## Save model:

```python
net.save("model.npz")
```

## Load model:

```python
net.load("model.npz")
```
