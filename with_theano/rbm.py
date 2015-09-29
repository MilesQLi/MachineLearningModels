#coding=utf-8

import numpy
import numpy as np
import theano
import gzip
import cPickle
import theano.tensor as T
import sys


class rbm(object):
    def __init__(self, n_visible, n_hidden):
        self.n_visible = n_visible
        self.n_hidden = n_hidden
        self.w = theano.shared(np.random.uniform(size=(n_visible, n_hidden)) , name = 'w')
        #TODO try size=(1, n_visible)
        self.bvis = theano.shared(np.random.uniform(size=(n_visible, )) , name = 'bvis')
        self.bhid = theano.shared(np.random.uniform(size=(n_hidden, )) , name = 'bhid')
        self.params = [self.bvis, self.bvis, self.bhid]
        
    def free_energy(self, x):
        vbias_term = T.dot(x, self.vbis)
        wx_b = self.bhid + T.dot(x,self.w)
        return - vbias_term - T.sum(T.log(1 + T.exp(vbias_term)), axis = 1)
    
    def 
    
    
    
    