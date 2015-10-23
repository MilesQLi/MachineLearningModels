#coding=utf-8
import numpy as np
from scipy.linalg import svd
from sklearn.datasets import fetch_olivetti_faces
from numpy.random import RandomState
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
import time

class knn(object):
    def __init__(self,k,dist = "Euclidean"):
        self.k = k
        if isinstance(dist,str):
            if hasattr(self,dist):
                self.dist = getattr(self,dist)
            else:
                print "Error: No that kind of distance function. Use Euclidean distance instead."
        else:
            self.dist = dist
    def Euclidean(x1,x2):
        return np.sqrt(np.sum((x1-x2)**2))
    def Manhattan(x1,x2):
        return np.sum(np.abs(x1-x2))
    
        
