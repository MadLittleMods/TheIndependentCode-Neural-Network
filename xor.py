import numpy as np
import matplotlib.pyplot as plt
#from mpl_toolkits.mplot3d import Axes3D
import logging

from dense import Dense
from activations import Sigmoid, Softmax
from losses import mse, mse_prime
from network import train, predict

X = np.reshape([[0, 0], [0, 1], [1, 0], [1, 1]], (4, 2, 1))
Y = np.reshape([[1, 0], [0, 1], [0, 1], [1, 0]], (4, 2, 1))

network = [
    Dense(2, 3),
    Sigmoid(),
    Dense(3, 2),
    Softmax()
]

# for layer in network:
#     if hasattr(layer, 'weights'):
#         logging.error("layer.weights %s %s", layer.weights.shape, layer.weights)
#     if hasattr(layer, 'bias'):
#         logging.error("layer.bias %s %s", layer.bias.shape, layer.bias)

# Match the initial weights and biases from our Zig implementaiton
network[0].weights = np.array([
  [3.251169104168574e-01, 1.0577112895890197e+00],
  [3.6170159819321346e-01, 1.4072284781826894e-02],
  [-1.271551261654049e+00, 3.2522080597322767e-01],
])
network[0].bias = np.array([
    [1.0e-01],
    [1.0e-01],
    [1.0e-01],
])
network[2].weights = np.array([
    [2.6545684575714984e-01, 8.636176515580903e-01, 2.953281182408529e-01],
    [1.1489972410202928e-02, -1.0382172576148672e+00, 2.655416761236997e-01],
])
network[2].bias = np.array([
    [1.0e-01],
    [1.0e-01],
])

# train
train(network, mse, mse_prime, X, Y, epochs=1, learning_rate=0.1)

# decision boundary plot
points = []
for x in np.linspace(0, 1, 20):
    for y in np.linspace(0, 1, 20):
        z = predict(network, [[x], [y]])
        # logging.error(f"network prediction for x={x} y={y} -> z={np.argmax(z)}")
        points.append([x, y, np.argmax(z)])

points = np.array(points)

fig = plt.figure()
ax = fig.add_subplot(111, projection="3d")
ax.scatter(points[:, 0], points[:, 1], points[:, 2], c=points[:, 2], cmap="winter")
# Show the plot for interactive use
plt.show()
# Save out the plot to a file so headless console people can see observe what's
# happening
plt.savefig("xor-graph.png")
