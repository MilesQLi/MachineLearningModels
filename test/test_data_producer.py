import sys
sys.path.append('..')
from datasets import *
import matplotlib.pyplot as plt
import numpy as np


def func(x):
    if 2*x[1] - 3*x[0] > 23.5:
        return 1
    else:
        return 0

if __name__ == '__main__':
    datas = dataset.classification_dataset_produce(func,[[5,6],[18,22]],10)
    x = [list(data[0]) for data in datas]
    y = [data[1] for data in datas]
    
    dataset.plot_data(x, y,[[5,6],[18,22]])