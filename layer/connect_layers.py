import numpy as np

class DenseLayer:
    def __init__(self, n_inputs, n_neurons):
        # n_inputs is the number of features from the previous layer
        # n_neurons is how many neurons this layer will have
        self.weights = np.random.randn(n_inputs, n_neurons) * 0.01
        self.biases = np.zeros((1, n_neurons))

    def forward(self, inputs):
        self.output = np.dot(inputs, self.weights) + self.biases
        self.inputs = inputs
        return self.output

    def backward(self, dvalues):
        # dvalues is the gradient from the next layer.
        # self.inputs was cached during the forward pass.

        self.dweights = np.dot(self.inputs.T, dvalues)
        self.dbiases = np.sum(dvalues, axis=0, keepdims=True)

        # Gradient on inputs (to be passed to the previous layer)
        self.dinputs = np.dot(dvalues, self.weights.T)