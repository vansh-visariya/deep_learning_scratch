class Model:
    def __init__(self):
        self.layers = []

    def add(self, layer):
        # Add a layer to the model sequence
        self.layers.append(layer)

    def forward(self, X):
        output = X
        for layer in self.layers:
            # Pass the output of the previous layer as input to the current layer
            output = layer.forward(output)
        return output
    
    def backward(self, y_pred, y_true):
        # Start with the loss function's backward pass
        # The first gradient comes from the loss function
        self.loss.backward(y_pred, y_true)
        dvalues = self.loss.dinputs

        for layer in reversed(self.layers):
            layer.backward(dvalues)
            # Update dvalues to be the gradient from the current layer
            dvalues = layer.dinputs

# update params
class Optimizer_SGD:
    def __init__(self, learning_rate=1.0):
        self.learning_rate = learning_rate

    def update_params(self, layer):
        layer.weights -= self.learning_rate * layer.dweights
        layer.biases -= self.learning_rate * layer.dbiases