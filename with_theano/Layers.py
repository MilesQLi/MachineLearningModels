#coding=utf-8
import theano
import theano.tensor as T
import numpy as np


class FullConnectedLayer(object):
    def  __init__(self, input, n_in, n_out):
        self.n_in = n_in
        self.n_out = n_out
        self.intput = input
        self.W = theano.shared(np.random.uniform(
            low=-np.sqrt(6. / (n_in + n_out)),
            high=np.sqrt(6. / (n_in + n_out)),
            size=(n_in, n_out)),name='W',borrow=True)
        self.b = theano.shared(np.random.uniform(
            low=-np.sqrt(6. / (n_in + n_out)),
            high=np.sqrt(6. / (n_in + n_out)),
            size=(n_out,)
            ),
            name='W',
            borrow=True)
        self.output = T.nnet.sigmoid(T.dot(input,self.W)+self.b)
        self.params = [self.W, self.b]