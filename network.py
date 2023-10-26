import logging

def predict(network, input):
    output = input
    for layer in network:
        output = layer.forward(output)
    return output

def train(network, loss, loss_prime, x_train, y_train, epochs = 1000, learning_rate = 0.01, verbose = True):
    for e in range(epochs):
        error = 0
        for x, y in zip(x_train, y_train):
            # forward
            output = predict(network, x)

            # error
            error += loss(y, output)

            # backward
            grad = loss_prime(y, output)
            layer_index = len(network) - 1
            for layer in reversed(network):
                logging.error("layer %s", layer_index)
                grad = layer.backward(grad, learning_rate)
                layer_index -= 1

        error /= len(x_train)
        if verbose:
            print(f"{e + 1}/{epochs}, error={error}")
