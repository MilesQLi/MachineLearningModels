import sys
sys.path.append('..')

import theano

from datasets import *
import matplotlib.pyplot as plt
import numpy as np
import theano.tensor as T
from with_theano import *



def func(x):
    if 2 * x[1] - 3 * x[0] > 23.5:
        return 1
    else:
        return 0

if __name__ == '__main__':
    datas = dataset.classification_dataset_produce(func, [[5, 6], [18, 22]], 1000)
    x = [list(data[0]) for data in datas]
    y = [data[1] for data in datas]
    
    linex1 = np.arange(3, 8, 0.3)
    linex2 = (23.5 + 3 * linex1) / 2
    assert(len(linex1 == linex2))
    
    trainx = np.array(x[:-30])
    trainy = np.array(y[:-30])
    # trainy.shape = (trainy.shape[0],1)
    testx = np.array(x[-30:])
    testy = np.array(y[-30:])
    
    X = T.matrix('x', dtype=theano.config.floatX)  # @UndefinedVariable
    Y = T.vector('y')
    
    logi = logistic_softmax_regression.LogisticRegression(X, 2)
    
    objective = 'cross_entropy'
    # objective = 'negative_log_likelihood'
    cost = getattr(logi, objective)(Y)
    
    
    '''
    adagrad sometimes gives perfect result
    '''
    #!!!!!!!!!!!
    # grads = [T.grad(cost, param) for param in logi.params]
    # updates = [(logi.W, logi.W - 0.01 * grads[0]), (logi.b, logi.b - 0.01 * grads[1])]
    # updates = utils.adagrad(cost, logi.params, 1.1)
    # updates = utils.rmsprop(cost, logi.params,0.052)
    # updates = utils.adadelta(cost, logi.params)
    updates = utils.gd(cost, logi.params, 0.012, 0, 0)
    
    train = theano.function([X, Y], cost, updates=updates)
    
    # input must be wrapped by []
    pred = theano.function([X], outputs=logi.y_pred)
    
    error = theano.function([X, Y], outputs=logi.error(Y))
    
    epoch = 5000

    for i in range(epoch):
        print 'epoch:', i, objective, ':', train(trainx, trainy), 'mean error:', error(testx, testy), '\r',
       # print np.array(pred(testx)).round().astype(int),testy
        # print logi.W.get_value()
        # print logi.b.get_value()
    print pred(testx)
    print testy
    print logi.W.get_value(), logi.b.get_value()
    
    xx, yy = np.meshgrid(np.arange(5, 6, 0.3),
                         np.arange(18, 22, 0.3))
    Z = pred(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    
    dataset.plot_data(x, y, [[5, 6], [18, 22]], (xx, yy, Z))
    
