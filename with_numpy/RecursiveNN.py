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

import matplotlib.pyplot as plt
import numpy as np


class RecursiveNN(object):
    def __init__(self,n_in,n_hid):
        self.n_in = n_in
        self.n_hid = n_hid
        self.wx = np.array(np.random.uniform(low=-0.012, high=0.012, size=(n_hid,n_in)))
        self.wh = np.array(np.random.uniform(low=-0.012, high=0.012, size=(n_hid,n_hid)))
        self.b = np.zeros(n_hid,)  # @UndefinedVariable
        
    def farward(self,x,h):
        return self.w.dot(x)+self.b
    
    def backward(self,delta_h,h):
        pass
    
if __name__ == '__main__':
    pass