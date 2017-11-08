import cPickle
import gzip
import sys
import time

from sklearn import datasets
from sklearn.datasets import *
import matplotlib.pyplot as plt
import numpy as np
sys.path.append('../')
from with_numpy import *


sys.path.append('..')
if __name__ == '__main__':
    f = gzip.open('mnist.pkl.gz', 'rb')
    train_set, valid_set, test_set = cPickle.load(f)
    train_x = train_set[0]
    train_y = train_set[1]
    x = train_x
    y = train_y
    valid_x = valid_set[0]
    valid_y = valid_set[1]
    fnn = FNN.FNN([50], 784, 10)
    # print fnn.pred(x)
    # print fnn.pred_prob(x)
    # print fnn.pred_prob(x,True)
    start = time.clock()
    epochs = 500
    best = 10.
    acc = 0
    n_samples = x.shape[0]
    fold = 5
    n_each_fold = n_samples / fold
    for epoch in range(epochs):
        for i in range(fold):
            fnn.train(x[i * n_each_fold:(i + 1) * n_each_fold], y[i * n_each_fold:(i + 1) * n_each_fold])
        error = fnn.error(valid_x, valid_y)
        if error <= best + 0.01:
            print 'epoch:', epoch, 'cross entropy:', fnn.cross_entropy(x, y), 'error:', error, '\r',
            best = error
            acc = 0
        else:
            acc += 1
            print 'best:', best, 'error:', error
            # if acc >= 50:
            #    break
        
    print 'epoch', epoch, 'total time:%.2fm' % ((time.clock() - start) / 60.)    

'''
if __name__ == '__main__':
    x,y = datasets.make_moons(1000, noise=0.21)
    fnn = FNN.FNN([3,3,3], 2, 2)
    #print fnn.pred(x)
    #print fnn.pred_prob(x)
    #print fnn.pred_prob(x,True)
    start = time.clock()
    epochs = 50000
    best = 10.
    acc = 0
    for epoch in range(epochs):
        for i in range(5):
            fnn.train(x[i*200:(i+1)*200],y[i*200:(i+1)*200])
        error = fnn.error(x, y)
        if error <= best + 0.01:
            print 'epoch:',epoch,'cross entropy:',fnn.cross_entropy(x, y),'error:',error,'\r',
            best = error
            acc = 0
        else:
            acc += 1
            print 'best:',best,'error:',error
            #if acc >= 50:
            #    break
        
    print 'epoch',epoch, 'total time:%.2fm'%((time.clock()-start) / 60.)
    
    
    xx, yy = np.meshgrid(np.arange(x[:,0].min()-.5, x[:,0].max()+.5, 0.3),
                         np.arange(x[:,1].min()-.5, x[:,1].max()+.5, 0.3))
    Z = fnn.pred(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    
    dataset.plot_data(x, y,[[x[:,0].min(), x[:,0].max()],[x[:,1].min(), x[:,1].max()]],(xx,yy,Z))

'''
