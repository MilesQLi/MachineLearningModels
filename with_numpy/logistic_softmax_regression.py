#coding=utf-8
import numpy as np



class LogisticRegression(object):
    def __init__(self,n_in):
        self.n_in = n_in
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
    
    def gradient_check(self, x, y):
        dw = self.gradient(x, y)
        calc_dw = np.zeros(self.n_in)
        delta = np.array(0.0001)
        for i in range(self.n_in):
            self.w[i] += delta
            pred = self.predict(x)
            result1 = (-y*np.log(pred)-(1-y)*np.log((1-pred))).mean()
            self.w[i] -= 2*delta
            pred = self.predict(x)
            result2 = (-y*np.log(pred)-(1-y)*np.log((1-pred))).mean()
            self.w[i] += delta
            calc_dw[i] = (result2-result1) / (2*delta)
        print 'calculated dw:'
        print dw
        print 'real dw:'
        print calc_dw
            
            
    
    
    def error(self,x,y):
        z = self.pred(x)
        z = z != y
        z = z.astype(int)
        return np.mean(z)