#coding=utf-8
import numpy as np
from scipy.linalg import svd
from sklearn.datasets import fetch_olivetti_faces
from numpy.random import RandomState
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
import time
import operator
import random
import csv
from matplotlib.colors import ListedColormap

#TODO kdtree version is still not implemented

class KNN(object):
    def __init__(self,datax,datay,k = 5,dist = "Euclidean",kdtree = True):
        self.k = k
        if isinstance(dist,str):
            if hasattr(self,dist):
                self.dist = getattr(self,dist)
            else:
                print "Error: No that kind of distance function. Use Euclidean distance instead."
        elif isinstance(dist,int):
            self.dist = self.krankdist
            self.dist_rank = dist
        else:
            self.dist = dist
        self.datax = datax
        self.datay = datay
        self.kdtree = kdtree
        if kdtree:
            #TODO
            pass
        
    def Euclidean(self, x1, x2):
        return np.sqrt(np.sum((x1-x2)**2))
    
    def Manhattan(self, x1, x2):
        return np.sum(np.abs(x1-x2))
    
    def krankdist(self, x1, x2):
        return (np.sum((x1-x2)**self.dist_rank))**(1/self.dist_rank)
    
    def find_kneighbours(self, x):
        if not self.kdtree:
            dists = []
            for i in range(len(self.datax)):
                dists.append((i,self.dist(x,self.datax[i])))
            dists.sort(key=operator.itemgetter(1))
            neighbours = []
            for i in range(self.k):
                neighbours.append(dists[i][0])
            return neighbours
    def predict(self, x):
        neighbours = self.find_kneighbours(x)
        counts = {}
        for i in neighbours:
            counts.setdefault(self.datay[i], 0)
            counts[self.datay[i]] += 1
        sortedcounts = sorted(counts.iteritems(), key=operator.itemgetter(1), reverse=True)
        return sortedcounts[0][0]
    def pred(self,X):
        result = []
        for i,x in zip(range(len(X)),X):
            print i,len(X),'done'
            result.append(self.predict(x))
        return np.array(result)

if __name__ == '__main__':
    n_neighbors = 5
    iris = load_iris()
    h = .02
    cmap_light = ListedColormap(['#FFAAAA', '#AAFFAA', '#AAAAFF'])
    cmap_bold = ListedColormap(['#FF0000', '#00FF00', '#0000FF'])
    X = iris.data[:, :2]  # we only take the first two features. We could
                      # avoid this ugly slicing by using a two-dim dataset
    y = iris.target
    knn = KNN(X, y, n_neighbors,'Euclidean',False)    
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))
    Z = knn.pred(np.c_[xx.ravel(), yy.ravel()])
    
    # Put the result into a color plot
    Z = Z.reshape(xx.shape)
    plt.figure()
    plt.pcolormesh(xx, yy, Z, cmap=cmap_light)
    # Plot also the training points
    plt.scatter(X[:, 0], X[:, 1], c=y, cmap=cmap_bold)
    plt.xlim(xx.min(), xx.max())
    plt.ylim(yy.min(), yy.max())
    plt.title("3-Class classification (k = %i)"% (n_neighbors))
    
    plt.show()