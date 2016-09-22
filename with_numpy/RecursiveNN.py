# coding=utf-8
import csv
import operator
import random
import time

from matplotlib.colors import ListedColormap
from numpy.random import RandomState
from scipy.linalg import svd
from sklearn import datasets
from sklearn.datasets import fetch_olivetti_faces
from sklearn.datasets import load_iris
import abc

import matplotlib.pyplot as plt
import numpy as np


class activation(object):
    def __init__(self):
        return
    
    @staticmethod
    def farward(x):
        raise NotImplementedError

    @staticmethod
    def backward(x):
        raise NotImplementedError

class identity(activation):
    def __init__(self):
        return
    @staticmethod
    def farward(x):
        return x
    @staticmethod
    def backward(x):
        return 1.


class sigmoid(activation):
    def __init__(self):
        return
    @staticmethod
    def farward(x):
        return 1. / (1. + np.exp(-x))
    @staticmethod
    def backward(x):
        t = 1. / (1. + np.exp(-x))
        return t*(1-t)


class RecursiveNN(object):
    def __init__(self,n_hid, activation = identity, wh1 = None, wh2 = None, b = None):
        self.n_hid = n_hid
        if wh1 == None:
            self.wh1 = np.array(np.random.uniform(low=-0.012, high=0.012, size=(n_hid,n_hid)))
        else:
            self.wh1 = wh1
        if wh2 == None:
            self.wh2 = np.array(np.random.uniform(low=-0.012, high=0.012, size=(n_hid,n_hid)))
        else:
            self.wh2 = wh2
        if b == None:
            self.b = np.zeros(n_hid,)  # @UndefinedVariable
        else:
            self.b = b
        self.activation = activation
        
    def farward(self,x1,x2):
        return self.activation.farward(self.wh1.dot(x1)+self.wh2.dot(x2)+self.b)
    
    def backward(self,delta_h, h, hlast1, hlast2):
        '''
        del_h is after activation
        it is not batch version yet
        '''
        tmp = delta_h*self.activation.backward(h)
        tmp = tmp.reshape(tmp.shape[0],1)
        hlast = hlast.reshape(hlast.shape[0],1)
        return [tmp.dot(hlast.T), tmp, tmp.dot(self.wh)]
    
    
if __name__ == '__main__':
    rnn = RecursiveNN(5,identity)
    x = np.random.random(5)
    x2 = np.random.random(5)
    print rnn.farward(x, x2)
    print rnn.backward(x, x2, x2)