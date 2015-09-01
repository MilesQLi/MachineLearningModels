#coding=utf-8
import theano
import theano.tensor as T
import numpy as np



class logisticRegression(object):
    #!!!!!!!!!!!!!!!!! n_out is the num of classes
    def __init__(self, input, n_in, n_out):
        np.random.RandomState(1234)
        self.intput = input
        self.W = theano.shared(np.random.uniform(high=np.sqrt(6. / (n_in + n_out)),size=(n_in, n_out)),name='W',borrow=True)
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
        
        
    def test_negative_log_likelihood(self,y):
        return self.y[T.arange(y.shape[0]), y]
    
    
    
    #!!!!!!!!!!!!!!!!!!!!!!!!!!1                           
    def negative_log_likelihood(self,y):
        return -T.mean(T.log(self.y)[T.arange(y.shape[0]), y])
    
    #!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    def error(self,y):
        return T.mean(T.neq(self.y_pred, y))