#coding=utf-8
import numpy as np
from scipy.linalg import svd
from sklearn.datasets import fetch_olivetti_faces
from numpy.random import RandomState
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn import datasets
import time
import operator
import random
import csv
from matplotlib.colors import ListedColormap

#TODO use_kdtree version is still not implemented


class node(object):
    def __init__(self,data=None,left=None,right=None,no = -1):
        self.data = data
        self.left = left
        self.right = right
        self.no = no

class neighbours(object):
    def __init__(self,target,k,dist):
        self.target = target
        self.k = k
        self.neighbours = []
        self.largest_distance = 0
        self.dist = dist
    def add(self,root):
        point = root.data
        dis = self.dist(point,self.target)
        for i,e in enumerate(self.neighbours):
            if i == self.k:
                return
            if e[1] > dis:
                self.neighbours.insert(i, [point,dis,root.no])
                self.neighbours = self.neighbours[:self.k]
                self.largest_distance = self.neighbours[-1][1]
                assert len(self.neighbours) <= self.k
                return
        self.neighbours.append([point,dis,root.no])
        self.neighbours = self.neighbours[:self.k]
        self.largest_distance = self.neighbours[-1][1]
        #print 'len:',len(self.neighbours)
        assert len(self.neighbours) <= self.k
    #return the indexs of neighbours
    def get_neighbours(self):
        return [e[2] for e in self.neighbours]

class kdtree(object):
    def __init__(self,X,dist):
        self.m = X.shape[1]
        self.n = X.shape[0]
        X_ = [(x,i) for i,x in enumerate(list(X))]
        self.root = self.build_tree(X_,0)
        self.dist = dist
    def build_tree(self,X,depth):
        if X == []:
            return None
        dim = depth%self.m
        X.sort(key = lambda x:x[0][dim])
        median = len(X) / 2
        return node(X[median][0],self.build_tree(X[:median], depth+1),self.build_tree(X[median+1:], depth+1),X[median][1])
    def search(self,target,k):
        def search_it(best_neighbours,root,depth):
            if root == None:
                return
            best_neighbours.add(root)
            dim = depth % self.m
            if target[dim] < root.data[dim]:
                near_part = root.left
                far_part = root.right
            else:
                near_part = root.right
                far_part = root.left
            search_it(best_neighbours, near_part, depth+1)
            
            if np.abs(root.data[dim]-target[dim]) < best_neighbours.largest_distance:
                search_it(best_neighbours, far_part, depth+1)
            
            
        best_neighbours = neighbours(target,k,self.dist)
        search_it(best_neighbours,self.root,0)
        return best_neighbours.get_neighbours()




class KNN(object):
    def __init__(self,datax,datay,k = 5,dist = "Euclidean",use_kdtree = True):
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
        self.use_kdtree = use_kdtree
        if use_kdtree:
            self.kdtree = kdtree(datax,self.dist)
        
    def Euclidean(self, x1, x2):
        return np.sqrt(np.sum((x1-x2)**2))
    
    def Manhattan(self, x1, x2):
        return np.sum(np.abs(x1-x2))
    
    def krankdist(self, x1, x2):
        return (np.sum((x1-x2)**self.dist_rank))**(1/self.dist_rank)
    
    def find_kneighbours(self, x):
        if not self.use_kdtree:
            dists = []
            for i in range(len(self.datax)):
                dists.append((i,self.dist(x,self.datax[i])))
            dists.sort(key=operator.itemgetter(1))
            neighbours = []
            for i in range(self.k):
                neighbours.append(dists[i][0])
            return neighbours
        else:
            return self.kdtree.search(x, self.k)
    #for compare with kd tree
    def print_kneighbours(self, x):
        if not self.use_kdtree:
            dists = []
            for i in range(len(self.datax)):
                dists.append((i,self.dist(x,self.datax[i])))
            dists.sort(key=operator.itemgetter(1))
            neighbours = []
            for i in range(self.k):
                neighbours.append(dists[i][0])
            return [self.datax[i] for i in neighbours]
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
            #print i,len(X),'done'
            result.append(self.predict(x))
        return np.array(result)
'''
if __name__ == '__main__':
    n_neighbors = 5
    X,y = datasets.make_moons(220,True,noise=0.21)
    knn = KNN(X, y, n_neighbors,'Euclidean',False) 
    kd = kdtree(X,knn.dist)
    
    for i in range(X.shape[0]):
        a = kd.search(X[i], n_neighbors)
        b = knn.find_kneighbours(X[i])
        assert(a==b)
        print a
        print b
        
'''


if __name__ == '__main__':
    n_neighbors = 5
    X,y = datasets.make_moons(22000,True,noise=0.21)
    test_begin = 15120
    t1 = time.clock()
    knn = KNN(X[:test_begin], y[:test_begin], n_neighbors,'Euclidean',True) 
    pred_y = knn.pred(X[test_begin:])
    length = len(pred_y)
    t2 = time.clock()
    #print pred_y
    #print y[120:]
    print 'Accuracy:',1 - np.sum(np.abs(pred_y-y[test_begin:]))/float(length),'time:',t2-t1
    t1 = time.clock()
    knn = KNN(X[:test_begin], y[:test_begin], n_neighbors,'Euclidean',False) 
    pred_y = knn.pred(X[test_begin:])
    length = len(pred_y)
    t2 = time.clock()
    #print pred_y
    #print y[120:]
    print 'Accuracy:',1 - np.sum(np.abs(pred_y-y[test_begin:]))/float(length),'time:',t2-t1

'''
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
    '''