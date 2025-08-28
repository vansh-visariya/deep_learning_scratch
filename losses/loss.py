import numpy as np

class Loss_CategoricalCrossentropy:
    def calculate(self, y_pred, y_true):
        # Clip predictions to prevent log(0) error
        y_pred_clipped = np.clip(y_pred, 1e-7, 1 - 1e-7)

        # Calculate the cross-entropy loss
        # For one-hot encoded labels, we only care about the log-probability of the correct class.
        correct_confidences = np.sum(y_pred_clipped * y_true, axis=1)
        negative_log_likelihoods = -np.log(correct_confidences)

        # Calculate the mean loss
        data_loss = np.mean(negative_log_likelihoods)
        return data_loss
    
    def backward(self, y_pred, y_true):

        num_samples = len(y_pred)
        num_labels = len(y_pred[0])

        # If labels are sparse, like [1, 0, 2], convert them to one-hot vectors
        if len(y_true.shape) == 1:
            y_true = np.eye(num_labels)[y_true]\
            
        # gradient
        self.dinputs = (y_pred - y_true) / num_samples

class Loss_BinaryCrossentropy:
    def calculate(self, y_pred, y_true):

        y_pred_clipped = np.clip(y_pred, 1e-7, 1 - 1e-7)

        sample_losses = -(y_true * np.log(y_pred_clipped) + (1 - y_true) * np.log(1 - y_pred_clipped))
        data_loss = np.mean(sample_losses)
        return data_loss
    
    def backward(self, y_pred, y_true):
        num_samples = len(y_pred)

        y_true = y_true.reshape(-1, 1)

        # gradient
        self.dinputs = (y_pred - y_true) / num_samples