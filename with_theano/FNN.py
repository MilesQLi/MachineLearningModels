# coding=utf-8
import theano

from Layers import *
from logistic_softmax_regression import *
import numpy as np
import theano.tensor as T


class FNN(object):
    def __init__(self, input, n_neurons, n_input, n_ouput):
        self.layers = []
        self.params = []
        layer = FullConnectedLayer(input, n_input, n_neurons[0])
        self.layers.append(layer)
        self.params += layer.params
        for j in range(1, len(n_neurons)):
            layer = FullConnectedLayer(T.nnet.sigmoid(self.layers[j - 1].output), n_neurons[j - 1], n_neurons[j])
            self.layers.append(layer)
            self.params += layer.params
        self.sigmoid = SoftmaxRegression(self.layers[-1].output, self.layers[-1].n_out, n_ouput)
        self.params += self.sigmoid.params
        self.y = self.sigmoid.y
        self.y_pred = self.sigmoid.y_pred
        self.negative_log_likelihood = self.sigmoid.negative_log_likelihood
        self.error = self.sigmoid.error
