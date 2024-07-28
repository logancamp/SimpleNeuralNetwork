from dense import Dense
from activations import Tanh
from losses import mse, mse_prime
import numpy as np
import pandas as pd



## initial training data
X = np.reshape([[0,0],[0,1],[1,0],[1,1]], (4,2,1)) # (len, array size x, array size y)
## results training data
Y = np.reshape([[0],[1],[1],[0]], (4,1,1))

## network shape
network = [
    Dense(2,4),
    Tanh(),
    Dense(4,1),
    Tanh()
]

## training
epochs = 100000 # trianing steps
learning_rate = 0.1

for e in range(epochs):
    error = 0
    for x, y in zip(X, Y):
        output = x
        for layer in network:
            output = layer.forward(output)

        error += mse(y, output)
        grad = mse_prime(y, output)
        for layer in reversed(network):
            grad = layer.backward(grad, learning_rate)

    error/= len(X)
    if (e + 1) % 100 == 0: # prints every 100
        print('%d/%d, error=%f'% (e+1, epochs, error))


## predictions
def predict(network, input_data):
    output = input_data
    for layer in network:
        output = layer.forward(output)
    return output

test_data = np.reshape([[0, 0], [0, 1], [1, 0], [1, 1]], (4, 2, 1))
predictions = [predict(network, x) for x in test_data]

print("Predictions:")
for i, (input_data, prediction) in enumerate(zip(test_data, predictions)):
    print(f"Input: {input_data.flatten()}, Prediction: {prediction.flatten()}")