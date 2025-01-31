# Neural Network From Scratch

This code is part of my video series on YouTube: [Neural Network from Scratch | Mathematics & Python Code](https://youtube.com/playlist?list=PLQ4osgQ7WN6PGnvt6tzLAVAEMsL3LBqpm).


## Setup

 1. Have Python 3.11+ installed
 1. Install [Poetry](https://python-poetry.org/docs/#installation)
 1. Install dependencies:
    ```sh
    poetry install
    ```

## Try it!

XOR example:

```sh
poetry run python xor.py
```

MNIST example:

```sh
poetry run python mnist.py
```

MNIST convolutional example:

```sh
poetry run python mnist_conv.py
```

## Example

```python
import numpy as np

from dense import Dense
from activations import Tanh
from losses import mse, mse_prime
from network import train

X = np.reshape([[0, 0], [0, 1], [1, 0], [1, 1]], (4, 2, 1))
Y = np.reshape([[0], [1], [1], [0]], (4, 1, 1))

network = [
    Dense(2, 3),
    Tanh(),
    Dense(3, 1),
    Tanh()
]

train(network, mse, mse_prime, X, Y, epochs=10000, learning_rate=0.1)
```
