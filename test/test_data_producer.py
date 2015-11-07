import sys

import theano

from datasets import *
import matplotlib.pyplot as plt
import numpy as np
import theano.tensor as T
from with_theano import *


sys.path.append('..')

def func(x):
    if 2 * x[1] - 3 * x[0] > 23.5:
        return 1
    else:
        return 0

if __name__ == '__main__':
    datas = dataset.classification_dataset_produce(func, [[5, 6], [18, 22]], 100)
    x = [list(data[0]) for data in datas]
    y = [data[1] for data in datas]
    
    linex1 = np.arange(3, 8, 0.3)
    linex2 = (23.5 + 3 * linex1) / 2
    assert(len(linex1 == linex2))
    
    dataset.plot_data(x, y, [[5, 6], [18, 22]], (linex1, linex2))
    
    
    
    
    
    
    
