import numpy as np


# -----------------------------------
# Convolution Layer
# -----------------------------------

class ConvLayer:

    def __init__(self, num_filters, filter_size):

        self.num_filters = num_filters
        self.filter_size = filter_size

        # Xavier Initialization
        self.filters = np.random.randn(
            num_filters,
            filter_size,
            filter_size
        ) * np.sqrt(2 / (filter_size * filter_size))

    def iterate_regions(self, image):

        h, w = image.shape

        for i in range(h - self.filter_size + 1):
            for j in range(w - self.filter_size + 1):

                region = image[
                    i:i + self.filter_size,
                    j:j + self.filter_size
                ]

                yield region, i, j

    def forward(self, input):

        self.last_input = input

        h, w = input.shape

        output = np.zeros((
            h - self.filter_size + 1,
            w - self.filter_size + 1,
            self.num_filters
        ))

        for region, i, j in self.iterate_regions(input):

            output[i, j] = np.sum(
                region * self.filters,
                axis=(1, 2)
            )

        return output

    def backward(self, d_L_d_out, learn_rate):

        d_L_d_filters = np.zeros(self.filters.shape)

        for region, i, j in self.iterate_regions(self.last_input):

            for f in range(self.num_filters):
                d_L_d_filters[f] += d_L_d_out[i, j, f] * region

        self.filters -= learn_rate * d_L_d_filters

        return None


# -----------------------------------
# ReLU Activation
# -----------------------------------

class ReLU:

    def forward(self, input):

        self.last_input = input

        return np.maximum(0, input)

    def backward(self, d_L_d_out):

        d_L_d_input = d_L_d_out.copy()

        d_L_d_input[self.last_input <= 0] = 0

        return d_L_d_input


# -----------------------------------
# Max Pool Layer
# -----------------------------------

class MaxPool:

    def iterate_regions(self, image):

        h, w, num_filters = image.shape

        new_h = h // 2
        new_w = w // 2

        for i in range(new_h):
            for j in range(new_w):

                region = image[
                    (i * 2):(i * 2 + 2),
                    (j * 2):(j * 2 + 2)
                ]

                yield region, i, j

    def forward(self, input):

        self.last_input = input

        h, w, num_filters = input.shape

        output = np.zeros((h // 2, w // 2, num_filters))

        for region, i, j in self.iterate_regions(input):

            output[i, j] = np.amax(region, axis=(0, 1))

        return output

    def backward(self, d_L_d_out):

        d_L_d_input = np.zeros(self.last_input.shape)

        for region, i, j in self.iterate_regions(self.last_input):

            h, w, f = region.shape

            amax = np.amax(region, axis=(0, 1))

            for i2 in range(h):
                for j2 in range(w):
                    for f2 in range(f):

                        if region[i2, j2, f2] == amax[f2]:

                            d_L_d_input[
                                i * 2 + i2,
                                j * 2 + j2,
                                f2
                            ] = d_L_d_out[i, j, f2]

        return d_L_d_input


# -----------------------------------
# Softmax Layer
# -----------------------------------

class Softmax:

    def __init__(self, input_len, nodes):

        self.weights = np.random.randn(
            input_len,
            nodes
        ) * np.sqrt(2 / input_len)

        self.biases = np.zeros(nodes)

    def forward(self, input):

        self.last_input_shape = input.shape

        input = input.flatten()

        self.last_input = input

        totals = np.dot(input, self.weights) + self.biases

        self.last_totals = totals

        exp = np.exp(totals - np.max(totals))

        return exp / np.sum(exp)

    def backward(self, d_L_d_out, learn_rate):

        for i, gradient in enumerate(d_L_d_out):

            if gradient == 0:
                continue

            t_exp = np.exp(self.last_totals - np.max(self.last_totals))

            S = np.sum(t_exp)

            d_out_d_t = -t_exp[i] * t_exp / (S ** 2)

            d_out_d_t[i] = (
                t_exp[i] * (S - t_exp[i])
            ) / (S ** 2)

            d_L_d_t = gradient * d_out_d_t

            d_L_d_w = (
                self.last_input[np.newaxis].T
                @ d_L_d_t[np.newaxis]
            )

            d_L_d_b = d_L_d_t

            d_L_d_inputs = self.weights @ d_L_d_t

            self.weights -= learn_rate * d_L_d_w

            self.biases -= learn_rate * d_L_d_b

            return d_L_d_inputs.reshape(
                self.last_input_shape
            )