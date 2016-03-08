import sys
import time
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
    
    X = T.fvector('x')  # @UndefinedVariable
    Y = T.iscalar('y')
    X2 = T.fmatrix('x2')  # @UndefinedVariable
    Y2 = T.ivector('y2')
    
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
    out =[cost]
    grads = T.grad(cost, logi.params)
    out.extend(grads)
    #updates = utils.gd(cost, logi.params, 0.012, 0, 0)
    
    train = theano.function([X, Y], out, allow_input_downcast=True)
    
    # input must be wrapped by []
    pred = theano.function([X2], outputs=logi.pred(X2),allow_input_downcast=True)
    
    error = theano.function([X2, Y2], outputs=logi.error(X2,Y2),allow_input_downcast=True)
    
    epoch = 200
    t1 =time.time()
    for i in range(epoch):
        for x,y in zip(trainx, trainy):
            train(x, y)
        print 'epoch:', i,'mean error:', error(testx, testy), '\r',
       # print np.array(pred(testx)).round().astype(int),testy
        # print logi.W.get_value()
        # print logi.b.get_value()
    print pred(testx)
    print testy
    print logi.W.get_value(), logi.b.get_value()
    print 'time:',time.time()-t1
    xx, yy = np.meshgrid(np.arange(5, 6, 0.3),
                         np.arange(18, 22, 0.3))
    Z = pred(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    
    #dataset.plot_data(x, y, [[5, 6], [18, 22]], (xx, yy, Z))
    
