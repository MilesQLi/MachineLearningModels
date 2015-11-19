# coding=utf-8
import theano

import numpy as np
import theano.tensor as T


class LogisticRegression(object):
    #!!!!!!!!!!!!!!!!! n_out is the num of classes
    def __init__(self, input, n_in):
        self.intput = input
        self.n_in = n_in
        
        #!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
        self.W = theano.shared(np.zeros((n_in,)), name='W', borrow=True)
        #!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
        # #!!  b mustbe a number not a vector or 
        self.b = theano.shared(np.array(0.), name='b')
        # self.y = 1 / (1 + T.exp(-T.dot(input, self.W) - self.b))
        self.y = T.nnet.sigmoid(T.dot(input, self.W) + self.b)
        self.y_pred = self.y > 0.5
        # self.y_pred = T.round(self.y)   works too
        
        self.params = [self.W, self.b]
        
    #!!!!!!!!!!!!!!!!!!!!!!!!!!1    
    '''
    def negative_log_likelihood(self,y):
        for i in T.arange(y.shape[0]):
            if y[i] == 0:
                self.y[i] = 1 - self.y[i]
            
        return -T.mean(T.log(self.y))
    
    def negative_log_likelihood(self,y):
        return -T.mean(T.log(self.y * (y*2.-1) + 3))
    '''                     
            
    def negative_log_likelihood(self, y):
        # TODO sometimes it is nan
        z = T.clip(T.abs_(y - 1 + self.y), 0.0000001, 0.999999999)
        return -T.mean(T.log(z))
        # return -T.mean(T.log(T.abs_(self.y - (1 - y))))   
    
    
    def cross_entropy(self, y):
        
        #return (-(y * T.log(self.y) + (1.0 - y) * T.log(1.0 - self.y))).mean()
        #return T.nnet.binary_crossentropy(self.y, y).mean()
         y_used = self.y
         y_used = T.clip(self.y, 0.0000001, 0.999999999)
         return T.mean(-y * T.log(y_used) - (1 - y) * T.log(1 - y_used))
    
    
    #!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    def error(self, y):
        return T.mean(T.neq(self.y_pred, y))

class SoftmaxRegression(object):
    #!!!!!!!!!!!!!!!!! n_out is the num of classes
    def __init__(self, input, n_in, n_out):
        self.intput = input
        self.n_in = n_in
        self.n_out = n_out
        self.W = theano.shared(np.zeros(size=(n_in, n_out)), name='W', borrow=True)
        self.b = theano.shared(np.zeros(size=(n_out,)), name='b', borrow=True)
        self.y = T.nnet.softmax(T.dot(input, self.W) + self.b)
        self.y_pred = T.argmax(self.y, axis=1)
        
        self.params = [self.W, self.b]
    
    #!!!!!!!!!!!!!!!!!!!!!!!!!!1                  
    def negative_log_likelihood(self, y):
        return -T.mean(T.log(self.y)[T.arange(y.shape[0]), y])
    
    def cross_entropy(self, y):
        return -T.mean(y * T.log(self.y)[T.arange(y.shape[0]), y])
    
    #!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    def error(self, y):
        return T.mean(T.neq(self.y_pred, y))
