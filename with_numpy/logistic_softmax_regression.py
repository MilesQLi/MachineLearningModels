#coding=utf-8
import numpy as np



class LogisticRegression(object):
    def __init__(self,n_in):
        self.w = np.random.uniform(low=-np.sqrt(6. / (n_in + 1)),high=np.sqrt(6. / (n_in + 1)),size=(n_in, ))
        #if [0] not [0.] and train with b+= then b will always be int type
        self.b = np.array([0.])
    
    def pred(self,x):
        y = 1 / (1+np.exp(-np.dot(x,self.w)-self.b))
        return (y > 0.5).astype(int)
    
    def predict(self,x):
        y = 1 / (1+np.exp(-np.dot(x,self.w)-self.b))
        return y  
    #def nll(self,x,y):
        
    def gradient(self,x,y):
        #print np.mean(((y-self.pred(x))*x.transpose()).transpose(),axis=0)
        return np.mean(((y-self.predict(x))*x.transpose()).transpose(),axis=0)
    
    def train(self,x,y, alpha):
        self.w += alpha* self.gradient(x, y)
        self.b += alpha* np.mean((y-self.predict(x)),axis = 0)
    
    def error(self,x,y):
        z = self.pred(x)
        z = z != y
        z = z.astype(int)
        return np.mean(z)