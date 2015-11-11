# coding=utf-8
import sys
sys.path.append('../../datasets/')
import cPickle
import gzip

import matplotlib.pyplot as plt
import numpy
import theano

import numpy as np
import theano.tensor as T
import reberGrammar
from sklearn import preprocessing
 
 
class rnn(object):
     def __init__(self, n_in, n_out, n_h, learning_rate=0.12):
        self.x = T.matrix(dtype=theano.config.floatX)  # @UndefinedVariable
        self.target = T.matrix(dtype=theano.config.floatX)  # @UndefinedVariable
        bound_x = numpy.sqrt(6. / (n_in + n_h))
        bound_h = numpy.sqrt(6. / (n_h + n_h))
        self.params = []
        self.w_x = theano.shared(np.array(np.random.uniform(low=-bound_x, high=bound_x, size=(n_in, n_h)), dtype=theano.config.floatX))  # @UndefinedVariable
        self.params.append(self.w_x)
        self.w_h = theano.shared(np.array(np.random.uniform(low=-bound_h, high=bound_h, size=(n_h, n_h)), dtype=theano.config.floatX))  # @UndefinedVariable
        self.params.append(self.w_h)
        self.b_h = theano.shared(np.array(np.random.uniform(low=-bound_h, high=bound_h, size=(n_h,)), dtype=theano.config.floatX))  # @UndefinedVariable
        self.params.append(self.b_h)
        self.w = theano.shared(np.array(np.random.uniform(low=-bound_h, high=bound_h, size=(n_h, n_out)), dtype=theano.config.floatX))  # @UndefinedVariable
        self.params.append(self.w)
        self.b = theano.shared(np.array(np.random.uniform(low=-bound_h, high=bound_h, size=(n_out,)), dtype=theano.config.floatX))  # @UndefinedVariable
        self.params.append(self.b)
        self.h0 = theano.shared(np.array(np.random.uniform(low=-bound_x, high=bound_x, size=(n_h,)), dtype=theano.config.floatX))  # @UndefinedVariable
        self.params.append(self.h0)
        
        def one_step(x, h1):
            h = T.nnet.sigmoid(T.dot(x, self.w_x) + T.dot(h1, self.w_h) + self.b_h)
            y = T.nnet.sigmoid(T.dot(h, self.w) + self.b)
            return h, y
        
        [hs, ys], _ = theano.scan(fn=one_step, sequences=self.x, outputs_info=[self.h0, None])
        cost = -T.mean(self.target * T.log(ys) + (1 - self.target) * T.log(1 - ys))
        grads = T.grad(cost, self.params)
        
        updates = [(param, param - learning_rate * grad) for param, grad in zip(self.params, grads)]
        
        self.train = theano.function([self.x, self.target], cost, updates=updates)
        
        self.predict = theano.function([self.x], ys)
        
if __name__ == '__main__':
    ls = rnn(7, 7, 10)
    train_data = reberGrammar.get_n_embedded_examples(1000)
    error = []
    for i in xrange(200):
        print '\n',i,'/200'
        err = 0
        for x,y in train_data:
            tmp = ls.train(x,y)
            err += tmp
            print tmp,'\r',
        error.append(err)
    plt.plot(np.arange(200), error, 'b-')
    plt.xlabel('epochs')
    plt.ylabel('error')
    plt.show()
    test_data = reberGrammar.get_n_embedded_examples(100)
    binarizer = preprocessing.Binarizer(threshold=0.1)
    error = 0
    for x,y in test_data:
        y_pred = ls.predict(x)
        y_pred = binarizer.transform(y_pred)
        for a,b in zip(y,y_pred):
            print '___________'
            print a
            print b
            error += np.mean(abs(a-b))
    print error       
        
        
        
