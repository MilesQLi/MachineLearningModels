import sys
sys.path.append('..')
from datasets import *
from with_theano import *
import matplotlib.pyplot as plt
import numpy as np
import theano
import theano.tensor as T
from sklearn import datasets


if __name__ == '__main__':
    x,y = datasets.make_moons(500, noise=0.21)
    y = y.astype(int)
    trainx = x[:-50]
    trainy = y[:-50]
    testx = x[-50:]
    testy = y[-50:]
    
    X = T.matrix('x',dtype=theano.config.floatX)  # @UndefinedVariable
    Y = T.ivector('y')
    
    logi = FNN.FNN(X,[5], 2,2)
    
    cost = logi.negative_log_likelihood(Y)
    
    grads = T.grad(cost,logi.params)
    
    alpha = 0.091
    
    #!!!!!!!!!!!
    updates = [(param, param - alpha * grad) for param,grad in zip(logi.params, grads)]
    
    train = theano.function([X,Y], logi.negative_log_likelihood(Y), updates=updates)
    
    #input must be wrapped by []
    pred = theano.function([X],outputs=logi.y_pred)
    
    
    error = theano.function([X,Y],outputs=logi.error(Y))
    
    epoch = 100000
    
    
    for i in range(epoch):
        print 'epoch:',i
        print 'neg log likelihood:',train(trainx,trainy)
        print 'mean error:',error(testx,testy)
       # print np.array(pred(testx)).round().astype(int),testy
        #print logi.W.get_value()
        #print logi.b.get_value()
    print pred(testx)
    print testy
    xx, yy = np.meshgrid(np.arange(x[:,0].min()-.5, x[:,0].max()+.5, 0.3),
                         np.arange(x[:,1].min()-.5, x[:,1].max()+.5, 0.3))
    Z = pred(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    
    dataset.plot_data(x, y,[[x[:,0].min(), x[:,0].max()],[x[:,1].min(), x[:,1].max()]],(xx,yy,Z))
    


'''
if __name__ == '__main__':
    n_samples = 5000
    n_features = 20
    n_informative = 10
    n_redundant = 0
    n_classes = 2
    n_clusters_per_class = 1
    x,y = datasets.make_classification(n_samples = n_samples, n_redundant = n_redundant,n_informative=n_informative,n_clusters_per_class = n_clusters_per_class, n_features = n_features,n_classes = n_classes)
    
    trainx = x[:-50]
    trainy = y[:-50]
    testx = x[-50:]
    testy = y[-50:]
    
    X = T.matrix('x',dtype=theano.config.floatX)  # @UndefinedVariable
    Y = T.ivector('y')
    
    logi = FNN.FNN(X,[20,10,10], 20,2)
    
    cost = logi.negative_log_likelihood(Y)
    
    grads = [T.grad(cost,param) for param in logi.params]
    
    alpha = 0.1
    
    #!!!!!!!!!!!
    updates = [(param, param - alpha * grad) for param,grad in zip(logi.params, grads)]
    
    train = theano.function([X,Y], logi.negative_log_likelihood(Y), updates=updates)
    
    #input must be wrapped by []
    pred = theano.function([X],outputs=logi.y_pred)
    
    
    error = theano.function([X,Y],outputs=logi.error(Y))
    
    epoch = 5000
    
    
    for i in range(epoch):
        print 'epoch:',i
        print 'neg log likelihood:',train(trainx,trainy)
        print 'mean error:',error(testx,testy)
       # print np.array(pred(testx)).round().astype(int),testy
        #print logi.W.get_value()
        #print logi.b.get_value()
    print pred(testx)
    print testy
    xx, yy = np.meshgrid(np.arange(x[:,0].min(), x[:,0].max(), 0.3),
                         np.arange(x[:,1].min(), x[:,1].max(), 0.3))
    Z = pred(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    
    dataset.plot_data(x, y,[[x[:,0].min(), x[:,0].max()],[x[:,1].min(), x[:,1].max()]],(xx,yy,Z))
    '''


'''
def func(x):
    if 2*x[1] - 3*x[0] > 23.5:
        return 1
    else:
        return 0


if __name__ == '__main__':
    datas = dataset.classification_dataset_produce(func,[[5,6],[18,22]],1000)
    x = [list(data[0]) for data in datas]
    y = [data[1] for data in datas]
    
    linex1 = np.arange(3,8,0.3)
    linex2 = (23.5+3*linex1)/2
    assert(len(linex1==linex2))
    
    trainx = np.array(x[:-20])
    trainy = np.array(y[:-20])
    testx = np.array(x[-20:])
    testy = np.array(y[-20:])
    
    X = T.matrix('x',dtype=theano.config.floatX)  # @UndefinedVariable
    Y = T.ivector('y')
    
    logi = FNN.FNN(X,[5,5], 2,2)
    
    cost = logi.negative_log_likelihood(Y)
    
    grads = [T.grad(cost,param) for param in logi.params]
    
    alpha = 0.1
    
    #!!!!!!!!!!!
    updates = [(param, param - alpha * grad) for param,grad in zip(logi.params, grads)]
    
    train = theano.function([X,Y], logi.negative_log_likelihood(Y), updates=updates)
    
    #input must be wrapped by []
    pred = theano.function([X],outputs=logi.y_pred)
    
    
    error = theano.function([X,Y],outputs=logi.error(Y))
    
    epoch = 30000
    
    
    for i in range(epoch):
        print 'epoch:',i
        print 'neg log likelihood:',train(trainx,trainy)
        print 'mean error:',error(testx,testy)
       # print np.array(pred(testx)).round().astype(int),testy
        #print logi.W.get_value()
        #print logi.b.get_value()
    print pred(testx)
    print testy
    
    xx, yy = np.meshgrid(np.arange(4, 7, 0.3),
                         np.arange(17, 23, 0.3))
    Z = pred(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    
    dataset.plot_data(x, y,[[5,6],[18,22]],(xx,yy,Z))

    
    '''
    