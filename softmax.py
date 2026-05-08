import numpy as np

class Softmax:

    def __init__(self, input_len, nodes):

        self.weights = np.random.randn(input_len, nodes) / input_len

        self.biases = np.zeros(nodes)

    def forward(self, input):

        self.last_input_shape = input.shape

        input = input.flatten()

        self.last_input = input

        totals = np.dot(input, self.weights) + self.biases

        self.last_totals = totals

        exp = np.exp(totals)

        return exp / np.sum(exp, axis=0)