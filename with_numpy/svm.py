#coding=utf-8
import numpy as np
from sklearn import datasets
import matplotlib.pyplot as plt
class svm(object):
    def __init__(self,C,kernel,epi=0.01):
        self.C = C
        self.epi = epi
        self.kernel = kernel
        self.b = 0
        
    def calcE(self,i):
        self.e[i] = 0
        for j in range(self.n):
            self.e[i] += self.alpha[j]*self.y[j]*self.k[i][j] + self.b
        self.e[i] -= self.y[j]
    def KKT(self,i):
        if self.alpha[i] == 0:
            return max(-self.y[i]*self.e[i] - self.epi, 0)
        if self.alpha[i] > 0 and self.alpha[i] < self.C:
            return np.abs(-self.y[i]*self.e[i] - self.epi)
        if self.alpha[i] == self.C:
            return max(self.y[i]*self.e[i] - self.epi, 0)
    
    def chooseA2(self,a1):
        maxi = -1
        maxi_i = -1
        #TODO　set threshold
        for i in range(self.n):
            temp = np.abs(self.e[a1] - self.e[i])
            if temp > maxi:
                maxi = temp
                maxi_i = i
        return maxi_i
    
    def updateAlpha(self,a1):
        a2 = self.chooseA2(a1)
        print 'a1:',a1,'a2',a2
        if self.y[a1] == self.y[a2]:
            l = max(0,self.alpha[a2]+self.alpha[a1]-self.C)
            h = min(self.C,self.alpha[a2]+self.alpha[a1])
        else:
            l = max(0,self.alpha[a2]-self.alpha[a1])
            h = min(self.C,self.alpha[a2]-self.alpha[a1]+self.C) 
        alpha2_old =  self.alpha[a2]
        alpha1_old =  self.alpha[a1]     
        self.alpha[a2] += self.y[a2] * (self.e[a1] - self.e[a2]) / (self.k[a1][a1]+self.k[a2][a2]+2*self.k[a1][a2])
        print 'alpha2:',self.alpha[a2]
        self.alpha[a2] = min(self.alpha[a2],h)
        self.alpha[a2] = max(self.alpha[a2],l)
        self.alpha[a1] += self.y[a1]*self.y[a2]*(alpha2_old-self.alpha[a2])
        b1 = -self.e[a1] - self.y[a1]*self.k[a1][a1]*(self.alpha[a1]-alpha1_old)-self.y[a2]*self.k[a2][a1]*(self.alpha[a2]-alpha2_old)+self.b
        b2 = -self.e[a2] - self.y[a1]*self.k[a1][a2]*(self.alpha[a1]-alpha1_old)-self.y[a2]*self.k[a2][a2]*(self.alpha[a2]-alpha2_old)+self.b
        if self.alpha[a1] > 0 and self.alpha[a1] < self.C:
            self.b = b1
        elif self.alpha[a2] > 0 and self.alpha[a2] < self.C:
            self.b = b2
        else:
            self.b = (b1+b2)/2
        self.calcE(a1)
        self.calcE(a2)
            
        
        
    def train(self,x,y,iter = 100):
        self.n = x.shape[0]
        self.alpha = np.zeros((self.n,),dtype = float)
        self.e = np.zeros((self.n,),dtype = float)
        self.x = x
        self.y = y
        self.k = np.zeros((self.n,self.n),dtype = float)
        for i in range(self.n):
            for j in range(i,self.n):
                self.k[i][j] = self.kernel(x[i],x[j])
                self.k[j][i] = self.k[i][j]
        print self.k
        for i in range(self.n):
            self.calcE(i)
        for i in range(iter):
            print 'iter:',i
            flag = False
            maxi = -1
            maxi_i = -1
            ran = np.nonzero(self.alpha)[0]
            for j in ran:
                temp = self.KKT(j)
                if temp > maxi:
                    maxi = temp
                    maxi_i = j
                if maxi > 0:
                    self.updateAlpha(maxi_i)
                    flag = True
            if not flag:
                for j in range(self.n):
                    temp = self.KKT(j) 
                    if temp > 0:
                       #print 'temp',temp
                       self.updateAlpha(j)  
                       break 
        self.sv = []
        for i in range(self.n):
            if self.alpha[i] > 0 and self.alpha[i] < self.C:
                self.sv.append(i)
    def predict(self,x):
        out = np.zeros(x.shape[0])
        for i in range(self.n):
            out += self.alpha[i]*self.y[i]*self.kernel(self.x[i],x)
        out += self.b
        print self.alpha
        print self.e
        #print self.y
        out2 = np.zeros(x.shape[0])
        out2[out>0] = 1
        out2[out<=0] = -1
        return out2
    def error(self,x,y):
        a = 0.
        a += x.shape[0]
        return np.sum(self.predict(x) != y) / a
    
def Gauss_kernel(x,z,sigma=1):
        return np.exp(-np.sum((x-z)**2)/(2*sigma**2))
    
def linear(x,z):  
    return np.sum(x*z)
        
if __name__ == '__main__':
    
    X,y = datasets.make_moons(100,noise=0.01)
    for i in range(len(y)):
        if y[i] == 0:
            y[i] = -1
    svm = svm(10,linear)
    svm.train(X,y,1000)
    #print svm.predict(X)
    plt.scatter(X[:,0],X[:,1], s=75, c=svm.predict(X), alpha=.5)    
    plt.show()   