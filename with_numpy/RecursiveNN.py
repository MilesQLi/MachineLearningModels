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
        pass
    
    @abc.absractmethod
    def farward(self):
        return

    @abc.absractmethod
    def backward(self):
        return



class RecursiveNN(object):
    def __init__(self,n_in,n_hid, wx = None, wh = None, b = None, activation = None):
        self.n_in = n_in
        self.n_hid = n_hid
        if wx == None:
            self.wx = np.array(np.random.uniform(low=-0.012, high=0.012, size=(n_hid,n_in)))
        else:
            self.wx = wx
        if wh == None:
            self.wh = np.array(np.random.uniform(low=-0.012, high=0.012, size=(n_hid,n_hid)))
        else:
            self.wh = wh
        if b == None:
            self.b = np.zeros(n_hid,)  # @UndefinedVariable
        else:
            self.b = b
        self.activation = activation
        
    def farward(self,x,h):
        return self.activation.farward(self.w.dot(x)+self.b)
    
    def backward(self,delta_h, h, hlast):
        '''
        del_h is after activation
        it is not batch version yet
        '''
        tmp = delta_h*self.activation.backward(h)
        tmp = tmp.reshape(tmp.shape[0],1)
        hlast = hlast.reshape(hlast.shape[0],1)
        return [tmp.dot(hlast.T), tmp, tmp.dot(self.wh)]
    
    
if __name__ == '__main__':
    pass