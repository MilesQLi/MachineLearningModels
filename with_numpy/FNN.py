# coding=utf-8
import numpy as np


class FNN(object):
    def __init__(self, n_neurons, n_input, n_ouput):
        self.Ws = []
        self.bs = []
        self.Ws.append(np.random.uniform(size=(n_input, n_neurons[0])))
        self.bs.append(np.random.uniform(size=(n_neurons[0],)))
        
        for i in range(1, len(n_neurons)):
            self.Ws.append(np.random.uniform(size=(n_neurons[i - 1], n_neurons[i])))
            self.bs.append(np.random.uniform(size=(n_neurons[i],)))
        self.Ws.append(np.random.uniform(size=(n_neurons[len(n_neurons) - 1], n_ouput)))
        self.bs.append(np.random.uniform(size=(n_ouput,)))
    
    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))
    
    def pred_prob(self, x , with_activations=False):
        activations = []
        activations.append(x)
        activation = self.sigmoid
        temp = activation(np.dot(x, self.Ws[0]) + self.bs[0])
        activations.append(temp)
        for i in range(1, len(self.Ws) - 1):
            temp = activation(np.dot(temp, self.Ws[i]) + self.bs[i])
            activations.append(temp)
        # print np.dot(temp, self.Ws[len(self.Ws)-1]).shape,self.bs[len(self.Ws)-1].shape
        temp = np.dot(temp, self.Ws[len(self.Ws) - 1]) + self.bs[len(self.Ws) - 1]
        exp = np.exp(temp)
        activations.append(exp / np.sum(exp, axis=1, keepdims=True))
        # print exp / np.sum(exp,axis = 1, keepdims = True)
        if with_activations == False:
            return activations[-1]
        else:
            return activations
            
    def pred(self, x):
        prob = self.pred_prob(x)
        return np.argmax(prob, axis=1)
    
    def error(self, x, y):
        y_pred = self.pred(x)
        z = y_pred != y
        z = z.astype(int)
        return np.mean(z)
    
    def cross_entropy(self, x, y):
        prob = self.pred_prob(x)
        return -np.mean(y * np.log(prob)[range(y.shape[0]), y])
        
    
    def train(self, x, y, alpha=0.121):
        n_sample = x.shape[0]
        activations = self.pred_prob(x, True)
        # print len(activations)
        delta = []
        dWs = []
        dbs = []
        activations[-1][range(n_sample), y] -= 1
        delta.append(activations[-1])
        # assert(activations[-1]==activations[2])
        ranges = range(len(self.Ws))
        ranges.reverse()
        # for i in activations:
            # print i.shape
        # print ranges
        
        # print activations[0].shape,activations[1].shape,activations[2].shape
        # print ranges
        for i in ranges:
            dWs.append(np.dot(activations[i].T, delta[-1]) / n_sample)
            dbs.append(np.sum(delta[-1], axis=0, keepdims=True) / n_sample)
            # print i, delta[-1].shape,self.Ws[i].T.shape,np.dot(delta[-1],self.Ws[i].T).shape,(activations[i]*(1-activations[i])).shape
            delta.append(np.dot(delta[-1], self.Ws[i].T) * activations[i] * (1 - activations[i]))
        dWs.reverse()
        dbs.reverse()
        # print len(self.Ws),len(self.bs),len(dWs),len(dbs)
        # print self.Ws[0].shape,dWs[0].shape
        # print self.Ws[1].shape,dWs[1].shape
        # print self.bs[0].shape,dbs[0].shape
        # print self.bs[1].shape,dbs[1].shape
        # print dWs[0][0]
        # print dWs[0]
        # print 'db0:',dbs[0]
        # print 'db1:',dbs[1]
        # print 'db2:',dbs[2]
        for i in range(len(self.Ws)):
            self. Ws[i] = self. Ws[i] - alpha * dWs[i]
            # print self. bs[i].shape,dbs[i].shape
            self. bs[i] = self. bs[i] - alpha * dbs[i]
    
