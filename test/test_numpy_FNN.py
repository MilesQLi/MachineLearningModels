import sys
sys.path.append('..')
from datasets import *
from with_numpy import *
import matplotlib.pyplot as plt
import numpy as np
from sklearn import datasets


if __name__ == '__main__':
    x,y = datasets.make_moons(6, noise=0.21)
    fnn = FNN.FNN([5], 2, 2)
    #print fnn.pred(x)
    #print fnn.pred_prob(x)
    #print fnn.pred_prob(x,True)
    fnn.train(x,y)
    
    