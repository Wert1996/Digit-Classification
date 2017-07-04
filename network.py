import numpy as np
import random


class Brain:
    def __init__(self, layers):
        self.layers = layers
        self.num_layers = len(layers)
        self.biases = [np.random.randn(layer, 1) for layer in layers[1:]]
        self.weights = [np.random.randn(pre, post) for pre, post in zip(layers[:-1], layers[1:])]

    def train(self, training_input, batch_size, epochs, eta, test_data):
        for epoch in range(epochs):
            training_input = random.shuffle(training_input)
            for i in range(len(training_input) - batch_size):
                tot_delta_w = [np.zeros(pre, post) for pre,post in zip(self.layers[:-1], self.layers[1:])]
                tot_delta_b = [np.zeros(layer, 1) for layer in self.layers[1:]]
                for x, y in training_input[i:i+batch_size]:
                    delta_w, delta_b = self.back_prop(x, y)
                    tot_delta_w = [td + cd for td, cd in zip(tot_delta_w, delta_w)]
                    tot_delta_b = [td + cd for td, cd in zip(tot_delta_b, delta_b)]
                self.weights = [w - (eta/batch_size)*tw for w, tw in zip(self.weights, tot_delta_w)]
                self.biases = [b - (eta/batch_size)*tb for b, tb in zip(self.biases, tot_delta_b)]
            if len(test_data) > 0:
                result = self.evaluate(test_data)
                print("Epoch {]: {}/{}".format(epoch, result, len(test_data)))

    def feed_forward(self, a):
        for i in range(self.num_layers-1):
            a = np.dot(a, self.weights[i]) + self.biases[i]
        return a

    def evaluate(self, test_data):
        eva = [y == np.argmax(x) for x, y in test_data]
        return np.sum(eva)

    def back_prop(self, x, y):
        activations = []
        z = []
        activations.append(x)
        delta_b = [np.zeros(layer, 1) for layer in self.layers[1:]]
        delta_w = [np.zeros(pre, post) for pre,post in zip(self.layers[:-1], self.layers[1:])]
        for i in range(self.num_layers - 1):
            z.append(np.dot(self.weights[i], activations[i])+self.biases[i])
            activations.append(self.sigmoid(z[i]))
        delta = self.cost_derivative(y, activations[-1])
        l = self.num_layers - 1
        while l > 0:
            delta_b[l-1] = delta
            delta_w[l-1] = np.dot(activations[l-1], delta)
            if l>= 2:
                delta = np.dot(np.transpose(delta_w[l-1]), delta)*self.sigmoid(z[l-2])
            l -= 1
        return delta_w, delta_b

    def sigmoid_derivative(self, z):
        return self.sigmoid(z)*(1.0-self.sigmoid(z))

    def sigmoid(self, z):
        return 1.0 / (1. + np.exp(-z))

    def cost_derivative(self, y_actual, fed):
        return fed - y_actual
