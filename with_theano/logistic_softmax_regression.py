#coding=utf-8
import theano
import theano.tensor as T
import numpy as np


class LogisticRegression(object):
    #!!!!!!!!!!!!!!!!! n_out is the num of classes
    def __init__(self, input, n_in):
        self.intput = input
        self.n_in = n_in
        
        #!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
        self.W = theano.shared(np.random.uniform(
            low=-np.sqrt(6. / (n_in + 1)),
            high=np.sqrt(6. / (n_in + 1)),
            size=(n_in, )),name='W',borrow=True)
        #!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
        ##!!  b mustbe a number not a vector or 
        self.b = theano.shared(np.random.uniform(),name='b')
        #self.y = 1 / (1 + T.exp(-T.dot(input, self.W) - self.b))
        self.y = T.nnet.sigmoid(T.dot(input,self.W)+self.b)
        self.y_pred = self.y > 0.5
        #self.y_pred = T.round(self.y)   works too
        
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
    
    def negative_log_likelihood(self,y):
        #TODO sometimes it is nan
        return -T.mean(T.log(T.abs_(self.y - (1 - y))))   
    
    
    def cross_entropy(self,y):
        return T.mean(-y * T.log(self.y) - (1-y) * T.log(1-self.y))
    
    
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
            low=-np.sqrt(6. / (n_in + n_out)),
            high=np.sqrt(6. / (n_in + n_out)),
            size=(n_in, n_out)),name='W',borrow=True)
        self.b = theano.shared(np.random.uniform(
            low=-np.sqrt(6. / (n_in + n_out)),
            high=np.sqrt(6. / (n_in + n_out)),
            size=(n_out,)
            ),
            name='b',
            borrow=True)
        self.y = T.nnet.softmax(T.dot(input,self.W)+self.b)
        self.y_pred = T.argmax(self.y,axis = 1)
        
        self.params = [self.W, self.b]
    
    #!!!!!!!!!!!!!!!!!!!!!!!!!!1                           
    def negative_log_likelihood(self,y):
        return -T.mean(T.log(self.y)[T.arange(y.shape[0]), y])
    
    def cross_entropy(self,y):
        return -T.mean(y*T.log(self.y)[T.arange(y.shape[0]), y])
    
    #!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    def error(self,y):
        return T.mean(T.neq(self.y_pred, y))