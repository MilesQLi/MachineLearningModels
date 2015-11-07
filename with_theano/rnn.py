# coding=utf-8

import cPickle
import gzip
import sys

import numpy
import theano

import numpy as np
import theano.tensor as T


def contextwin(x, n_win):
    l = list(x)
    pad = n_win // 2 * [-1] + l + n_win // 2 * [-1]
    return [pad[i:i + n_win] for i in range(len(l))]


class rnn(object):
    '''
        nh :: dimension of the hidden layer
        nc :: number of classes
        ne :: number of word embeddings in the vocabulary
        de :: dimension of the word embeddings
        cs :: word window context size
        lr :: learning rate
    '''
    def __init__(self, nh, nc, ne, de, cs, lr):
        self.wx = theano.shared(np.random.uniform(size=(de * cs, nh)) , name='wh')
        self.wh = theano.shared(np.random.uniform(size=(nh, nh)) , name='wh')
        self.bh = theano.shared(np.random.uniform(size=(nh,)), name='bh')
        self.w = theano.shared(np.random.uniform(size=(nh, nc)) , name='w')
        self.b = theano.shared(np.random.uniform(size=(nc,)) , name='b')
        self.h0 = theano.shared(np.random.uniform(size=(nh,)) , name='b')
        self.emb = theano.shared(np.random.uniform(size=(ne + 1, de)) , name='emb')
        
        self.params = [self.wx, self.wh, self.bh, self.w, self.b, self.h0, self.emb]
        
        self.idxs = T.imatrix()
        self.x = self.emb[self.idxs].reshape((self.idxs.shape[0], cs * de))
        self.y_real = T.ivector()
        
    
        def recurrence(x, h1):
            h0 = T.nnet.sigmoid(T.dot(x, self.wx) + T.dot(h1, self.wh) + self.bh)
            s0 = T.nnet.softmax(T.dot(h0, self.w) + self.b)
            return [h0, s0]
        
        [h, s], _ = theano.scan(recurrence, sequences=[self.x], outputs_info=[self.h0, None])
        
        self.y = s[:, 0, :]
        self.y_pred = T.argmax(self.y, axis=1)
        self.neg_log_likelihood = -T.mean(T.log(self.y)[T.arange(self.y_real.shape[0]), self.y_real])
        error = T.mean(T.neq(self.y_pred, self.y_real))
        
        self.grads = T.grad(self.neg_log_likelihood, self.params)
        updates = [(param, param - lr * grad) for param, grad in zip(self.params, self.grads)]
        
        self.train = theano.function([self.idxs, self.y_real], self.neg_log_likelihood, updates=updates)
        self.pred = theano.function([self.idxs], self.y_pred)
        self.error = theano.function([self.idxs, self.y_real], error)
        
        self.aa = theano.function([self.idxs], s)

            

def test_atis():
    filename = 'atis.fold3.pkl.gz'
    epoch = 50
    f = gzip.open(filename, 'rb')
    train_set, valid_set, test_set, dicts = cPickle.load(f)
    train_x = train_set[0]
    train_y = train_set[2]
    valid_x = valid_set[0]
    valid_y = valid_set[2]
    test_x = test_set[0]
    test_y = test_set[2]
    
    nc = len(set(reduce(lambda x, y:list(x) + list(y), train_y + valid_y + test_y)))
    ne = len(set(reduce(lambda x, y:list(x) + list(y), train_x + valid_x + test_x)))
    
    word22idx = dicts['words2idx']
    label2idx = dicts['labels2idx']

    idx2words = dict((v, k) for k, v in word22idx.iteritems())
    idx2label = dict((v, k) for k, v in label2idx.iteritems())
    nh = 50
    de = 50
    cs = 5
    lr = 0.0821
    
    rn = rnn(nh, nc, ne, de, cs, lr)
    for i in range(epoch):
        print '\nepoch:%d' % i
        sys.stdout.flush()
        length = len(train_x)
        nlls = []
        for x, y, j in zip(train_x, train_y, range(length)):
            nll = rn.train(contextwin(x, cs), y)
            nlls.append(nll)
            print 'sample:%d/%d\r' % (j, length),
            sys.stdout.flush()
        nlls = np.array(nlls)
        print 'nll:', nlls.mean()
        length = len(valid_x)
        valid_error = []
        for x, y, j in zip(valid_x, valid_y, range(length)):
           # print rn.pred(x)
            valid_error.append(1)
            
        valid_error = np.array(valid_error)
        print 'valid error:', valid_error.mean()
    
    
if __name__ == '__main__':
    # a = [1,2,3,4,5,6]
    # print contextwin(a, 3)
    test_atis()
    
    
