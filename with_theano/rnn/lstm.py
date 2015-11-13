# coding=utf-8

import cPickle
import gzip
from sklearn import preprocessing
import sys
import time

sys.path.append('../../datasets/')
sys.path.append('../')
import numpy
import theano

import matplotlib.pyplot as plt
import numpy as np
import reberGrammar
import theano.tensor as T

import utils




class lstm(object):
    def __init__(self, n_in, n_out, n_h, learning_rate=1.12):
        
        self.x = T.matrix(dtype=theano.config.floatX)  # @UndefinedVariable
        self.target = T.matrix(dtype=theano.config.floatX)  # @UndefinedVariable
        
        bound_x = numpy.sqrt(6. / (n_in + n_h))
        bound_h = numpy.sqrt(6. / (n_h + n_h))
        self.params = []
        
        self.w_xi = theano.shared(np.array(np.random.uniform(low=-bound_x, high=bound_x, size=(n_in, n_h)), dtype=theano.config.floatX))  # @UndefinedVariable
        self.params.append(self.w_xi)
        self.w_hi = theano.shared(np.array(np.random.uniform(low=-bound_h, high=bound_h, size=(n_h, n_h)), dtype=theano.config.floatX))  # @UndefinedVariable
        self.params.append(self.w_hi)
        self.w_ci = theano.shared(np.array(np.random.uniform(low=-bound_h, high=bound_h, size=(n_h, n_h)), dtype=theano.config.floatX))  # @UndefinedVariable
        self.params.append(self.w_ci)
        self.b_i = theano.shared(np.array(np.random.uniform(low=-bound_h, high=bound_h, size=(n_h,)), dtype=theano.config.floatX))  # @UndefinedVariable
        self.params.append(self.b_i)
        
        self.w_xf = theano.shared(np.array(np.random.uniform(low=-bound_x, high=bound_x, size=(n_in, n_h)), dtype=theano.config.floatX))  # @UndefinedVariable
        self.params.append(self.w_xf)
        self.w_hf = theano.shared(np.array(np.random.uniform(low=-bound_h, high=bound_h, size=(n_h, n_h)), dtype=theano.config.floatX))  # @UndefinedVariable
        self.params.append(self.w_hf)
        self.w_cf = theano.shared(np.array(np.random.uniform(low=-bound_h, high=bound_h, size=(n_h, n_h)), dtype=theano.config.floatX))  # @UndefinedVariable
        self.params.append(self.w_cf)
        self.b_f = theano.shared(np.array(np.random.uniform(low=-bound_h, high=bound_h, size=(n_h,)), dtype=theano.config.floatX))  # @UndefinedVariable
        self.params.append(self.b_f)
        
        self.w_xc = theano.shared(np.array(np.random.uniform(low=-bound_x, high=bound_x, size=(n_in, n_h)), dtype=theano.config.floatX))  # @UndefinedVariable
        self.params.append(self.w_xc)
        self.w_hc = theano.shared(np.array(np.random.uniform(low=-bound_h, high=bound_h, size=(n_h, n_h)), dtype=theano.config.floatX))  # @UndefinedVariable
        self.params.append(self.w_hc)
        self.b_c = theano.shared(np.array(np.random.uniform(low=-bound_h, high=bound_h, size=(n_h,)), dtype=theano.config.floatX))  # @UndefinedVariable       
        self.params.append(self.b_c)
        
        self.w_xo = theano.shared(np.array(np.random.uniform(low=-bound_x, high=bound_x, size=(n_in, n_h)), dtype=theano.config.floatX))  # @UndefinedVariable
        self.params.append(self.w_xo)
        self.w_ho = theano.shared(np.array(np.random.uniform(low=-bound_h, high=bound_h, size=(n_h, n_h)), dtype=theano.config.floatX))  # @UndefinedVariable
        self.params.append(self.w_ho)
        self.w_co = theano.shared(np.array(np.random.uniform(low=-bound_h, high=bound_h, size=(n_h, n_h)), dtype=theano.config.floatX))  # @UndefinedVariable
        self.params.append(self.w_co)
        self.b_o = theano.shared(np.array(np.random.uniform(low=-bound_h, high=bound_h, size=(n_h,)), dtype=theano.config.floatX))  # @UndefinedVariable
        self.params.append(self.b_o)
        
        self.w_y = theano.shared(np.array(np.random.uniform(low=-bound_x, high=bound_x, size=(n_h, n_out)), dtype=theano.config.floatX))  # @UndefinedVariable
        self.params.append(self.w_y)
        self.b_y = theano.shared(np.array(np.random.uniform(low=-bound_h, high=bound_h, size=(n_out,)), dtype=theano.config.floatX))  # @UndefinedVariable
        self.params.append(self.b_y)
        
        self.c0 = theano.shared(np.array(np.random.uniform(low=-bound_x, high=bound_x, size=(n_h,)), dtype=theano.config.floatX))  # @UndefinedVariable
        self.params.append(self.c0)
        self.h0 = T.tanh(self.c0)
        
        
        
        def one_step(x_t, h_t1, c_t1):
            i = T.nnet.sigmoid(T.dot(x_t, self.w_xi) + T.dot(h_t1, self.w_hi) + T.dot(c_t1, self.w_ci) + self.b_i)
            f = T.nnet.sigmoid(T.dot(x_t, self.w_xf) + T.dot(h_t1, self.w_hf) + T.dot(c_t1, self.w_cf) + self.b_f)
            o = T.nnet.sigmoid(T.dot(x_t, self.w_xo) + T.dot(h_t1, self.w_ho) + T.dot(c_t1, self.w_co) + self.b_o)
            c = f * c_t1 + i * T.tanh(T.dot(x_t, self.w_xc) + T.dot(h_t1, self.w_hc) + self.b_c)
            h = o * T.tanh(c)
            y = T.nnet.sigmoid(T.dot(h, self.w_y) + self.b_y)
            return [c, h, y]
        
        [_, hs, ys], _ = theano.scan(fn=one_step, sequences=self.x, outputs_info=[self.c0, self.h0, None])
        
        cost = -T.mean(self.target * T.log(ys) + (1 - self.target) * T.log(1 - ys))
        

         
        #updates = utils.rmsprop(cost, self.params, learning_rate)
        #updates = utils.adadelta(cost, self.params)       
        updates = utils.gd(cost, self.params, learning_rate)
        #updates = utils.adagrad(cost, self.params, learning_rate)
        #updates = utils.gd_momentum(cost, self.params, learning_rate)
        #updates = utils.adam(cost, self.params)
        
        
        self.train = theano.function([self.x, self.target], cost, updates=updates)
        
        self.predict = theano.function([self.x], ys)
        
if __name__ == '__main__':
    ls = lstm(7, 7, 10)
    train_data = reberGrammar.get_n_embedded_examples(1000)
    error = []
    t1 = time.clock()
    for i in xrange(60):
        print '\n', i, '/60'
        err = 0
        for x, y in train_data:
            tmp = ls.train(x, y)
            err += tmp
            print tmp, '\r',
        #print ls.predict(train_data[0][0])
        error.append(err)
    print 'time:', time.clock() - t1
    plt.plot(np.arange(60), error, 'b-')
    plt.xlabel('epochs')
    plt.ylabel('error')
    plt.show()
    print error
    test_data = reberGrammar.get_n_embedded_examples(100)
    binarizer = preprocessing.Binarizer(threshold=0.1)
    error = 0
    for x, y in test_data:
        y_pred = ls.predict(x)
        #print y_pred
        y_pred = binarizer.transform(y_pred)
        for a, b in zip(y, y_pred):
            error += np.mean(a - b)
    print error
