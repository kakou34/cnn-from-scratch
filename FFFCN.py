import copy

import numpy as np
from scipy import special

from Errors import MatrixDimensionError


# Class for single layer feed forward fully connected network
# Activation function: softmax, loss function: -ln(yc)
class FFFCN:

    # initialization
    def __init__(self, lr, in_s, op_s):

        # initialize weights using He initialization
        self.weights = np.random.randn(op_s, in_s + 1) * np.sqrt(2 / in_s)
        self.lr = lr
        self.input_size = in_s
        self.output_size = op_s
        self.initial = ()

    # softmax activation function
    def softmax(self, v):
        return special.expit(v) / np.sum(special.expit(v))

    # feed-forward operation
    def predict(self, input_vec):
        # adding a leading 1 to the input vector for the bias
        input_vec = np.insert(input_vec, 0, 1., axis=0)
        if len(input_vec) != self.weights.shape[1]:
            raise MatrixDimensionError('FC input', 'FC weights')
        # v = W.x
        v = np.dot(self.weights, input_vec)
        # output = softmax(v)
        output = self.softmax(v)
        return output

    def backpropagation(self, input_vec, yc_idx):
        # insert 1 to the beginning of input tor for bias
        input_vec = np.insert(input_vec, 0, 1., axis=0)
        # check for matrix multiplication
        if len(input_vec) != self.weights.shape[1]:
            raise MatrixDimensionError('FC input', 'FC weights')
        # get the v values
        v = np.dot(self.weights, input_vec)
        # get the outputs using softmax transfer function
        output = special.softmax(v)
        # Calculating delta values for each output (d(error)/d(vi))
        # for incorrect output delta is = output (for error = -ln(yc))
        deltas_out = copy.deepcopy(output)
        # for the expected output delta = output found - 1
        deltas_out[yc_idx] = output[yc_idx] - 1
        # finding the local gradients for inputs
        deltas_in = np.dot(np.transpose(self.weights), deltas_out)
        # updating the weights and bias for FC
        curr_weights = copy.deepcopy(self.weights)
        del_weight = np.dot(deltas_out, np.transpose(input_vec))
        self.weights = curr_weights - del_weight * self.lr
        # return the input local gradients for CNN backpropagation
        return deltas_in
