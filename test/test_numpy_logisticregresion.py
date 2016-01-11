import sys

sys.path.append('..')

from datasets import *
import matplotlib.pyplot as plt
import numpy as np
from with_numpy import *


def func(x):
    if 2 * x[1] - 3 * x[0] > 23.5:
        return 1
    else:
        return 0

if __name__ == '__main__':
    datas = dataset.classification_dataset_produce(func, [[5, 6], [18, 22]], 1100)
    x = [list(data[0]) for data in datas]
    y = [data[1] for data in datas]
    
    linex1 = np.arange(3, 8, 0.3)
    linex2 = (23.5 + 3 * linex1) / 2
    assert(len(linex1 == linex2))
    
    trainx = np.array(x[:-100])
    trainy = np.array(y[:-100])
    # trainy.shape = (trainy.shape[0],1)
    testx = np.array(x[-100:])
    testy = np.array(y[-100:])
    
    logi = logistic_softmax_regression.LogisticRegression(2)
    
    epoch = 500
    alpha = 1.162
    for k in range(5):
        alpha /= 10
        for i in range(epoch):
            for j in range(5):
                logi.train(trainx[j * 200:(j + 1) * 200], trainy[j * 200:(j + 1) * 200], alpha)
            print 'epoch:', i + k * epoch, 'error:%10f' % logi.error(testx, testy), logi.gradient(trainx, trainy), np.mean((trainy - logi.pred(trainx)), axis=0) 
    print logi.w
    print logi.b
    print testy
    print logi.pred(testx)
    print logi.error(testx, testy)
    xx, yy = np.meshgrid(np.arange(5, 6, 0.3),
                         np.arange(18, 22, 0.3))
    Z = logi.pred(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    
    logi.gradient_check(trainx[1 * 200:(1 + 1) * 200], trainy[1 * 200:(1 + 1) * 200])
    
    learned_x = np.arange(3, 8, 0.3)
    learned_y = (-logi.b - logi.w[0] * learned_x) / logi.w[1]
    
    origin_x = np.arange(3, 8, 0.3)
    origin_y = (23.5 + 3 * origin_x) / 2
    
    dataset.plot_data(x, y, [[5, 6], [18, 22]], (xx, yy, Z), (origin_x, origin_y), (learned_x, learned_y))
