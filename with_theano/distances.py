# coding=utf-8

import cPickle
import gzip
from sklearn import preprocessing
import sys
import time

import numpy
import theano

import matplotlib.pyplot as plt
import numpy as np
import theano.tensor as T

def cos(x1,x2):
    x = (x1*x2).sum(axis=1)
    x3=x/(T.sqrt((x1**2).sum(axis=1))*T.sqrt((x2**2).sum(axis=1)))
    return x3

def euro_dist(x1,x2):
    x = T.sqrt(((x1-x2)**2).sum())
    return x