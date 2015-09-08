import sys
sys.path.append('..')
from datasets import *
from with_numpy import *
import matplotlib.pyplot as plt
import numpy as np
from sklearn import datasets
import time


if __name__ == '__main__':
    x,y = datasets.make_moons(1000, noise=0.21)
    fnn = FNN.FNN([5], 2, 2)
    #print fnn.pred(x)
    #print fnn.pred_prob(x)
    #print fnn.pred_prob(x,True)
    start = time.clock()
    epoch = 50000
    for i in range(epoch):
        fnn.train(x,y)
        #print 'epoch:',i,'error:',fnn.error(x, y),'\r',
        
    print start,time.clock()
    print 'epoch',epoch, 'total time:%.2fm'%((time.clock()-start) / 60.)
    