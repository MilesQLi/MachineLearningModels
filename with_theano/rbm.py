#coding=utf-8

import numpy
import numpy as np
import theano
import gzip
import cPickle
import theano.tensor as T
import sys
from theano.tensor.shared_randomstreams import RandomStreams


class rbm(object):
    def __init__(self, n_visible, n_hidden):
        self.n_visible = n_visible
        self.n_hidden = n_hidden
        self.w = theano.shared(np.random.uniform(size=(n_visible, n_hidden)) , name = 'w')
        #TODO try size=(1, n_visible)
        self.bvis = theano.shared(np.random.uniform(size=(n_visible, )) , name = 'bvis')
        self.bhid = theano.shared(np.random.uniform(size=(n_hidden, )) , name = 'bhid')
        self.params = [self.bvis, self.bvis, self.bhid]
        theano_rng = RandomStreams()
        
    def free_energy(self, x):
        vbias_term = T.dot(x, self.vbis)
        wx_b = self.bhid + T.dot(x,self.w)
        return - vbias_term - T.sum(T.log(1 + T.exp(vbias_term)), axis = 1)
    
    def propup(self, x):
        z = T.dot(x, self.w)+self.bhid
        return (z,T.nnet.sigmoid(z))
    
    def sample_h_given_v(self, x):
        z, a = self.propup(x)
        sample = self.theano_rng.binomial(size=a.shape, n=1, p = a, dtype=theano.config.floatX)  # @UndefinedVariable
        return (z, a, sample)
    
    def propdown(self, y):
        z = T.dot(y, self.w.T) + self.bvis
        return (z, T.nnet.sigmoid(z))
    
    def sample_v_given_h(self, y):
        z, x = self.propdown(y)
        sample = self.theano_rng.binomial(size=x.shape, n=1, p = x, dtype=theano.config.floatX)  # @UndefinedVariable
        return (z, x, sample)
    
    def gibbs_vhv(self, x):
        z1, a1, sampley1 = self.sample_h_given_v(x)
        z2, x1, samplex1 = self.sample_v_given_h(sampley1)
        return (z1,a1,sampley1,z2,x1,samplex1)
    
    def gibbs_hvh(self, y):
        z1, x1, samplex1 = self.sample_v_given_h(y)
        z2, y1, sampley1 = self.sample_h_given_v(samplex1)
        return (z1, x1, samplex1, z2, y1, sampley1)
    