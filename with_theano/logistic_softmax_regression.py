#coding=utf-8
import theano
import theano.tensor as T
import numpy as np


class LogisticRegression(object):
    #!!!!!!!!!!!!!!!!! n_out is the num of classes
    def __init__(self, input, n_in):
        self.intput = input
        self.n_in = n_in
        self.W = theano.shared(np.random.uniform(
            high=np.sqrt(6. / (n_in + 1)),
            size=(n_in, 1)),name='W',borrow=True)
        self.b = theano.shared(np.random.uniform(
            low=-np.sqrt(6. / (n_in + 1)),
            high=np.sqrt(6. / (n_in + 1)),
            size=(1,)
            ),
            name='W',
            borrow=True)
        self.y = T.nnet.sigmoid(T.dot(input,self.W)+self.b)
        self.y_pred = T.round(self.y)
        
        self.params = [self.W, self.b]
        
    #!!!!!!!!!!!!!!!!!!!!!!!!!!1    
    '''
    def negative_log_likelihood(self,y):
        for i in T.arange(y.shape[0]):
            if y[i] == 0:
                self.y[i] = 1 - self.y[i]
            
        return -T.mean(T.log(self.y))
    '''                       
    def negative_log_likelihood(self,y):
        self.y = self.y * (y*2-1)
        return -T.mean(T.log(self.y))
    
    #!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    def error(self,y):
        return T.mean(T.neq(self.y_pred, y))

class SoftmaxRegression(object):
    #!!!!!!!!!!!!!!!!! n_out is the num of classes
    def __init__(self, input, n_in, n_out):
        self.intput = input
        self.n_in = n_in
        self.n_out = n_out
        self.W = theano.shared(np.random.uniform(
            high=np.sqrt(6. / (n_in + n_out)),
            size=(n_in, n_out)),name='W',borrow=True)
        self.b = theano.shared(np.random.uniform(
            low=-np.sqrt(6. / (n_in + n_out)),
            high=np.sqrt(6. / (n_in + n_out)),
            size=(n_out,)
            ),
            name='W',
            borrow=True)
        self.y = T.nnet.softmax(T.dot(input,self.W)+self.b)
        self.y_pred = T.argmax(self.y,axis = 1)
        
        self.params = [self.W, self.b]
    
    #!!!!!!!!!!!!!!!!!!!!!!!!!!1                           
    def negative_log_likelihood(self,y):
        return -T.mean(T.log(self.y)[T.arange(y.shape[0]), y])
    
    #!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    def error(self,y):
        return T.mean(T.neq(self.y_pred, y))