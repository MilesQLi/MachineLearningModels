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
        temp = 0
        for j in range(self.n):
            temp += self.alpha[j]*self.y[j]*self.k[i][j]
        self.e[i] = temp + self.b - self.y[i]
        #print 'ei',self.e[i]
            
    def KKT(self,i):
        if self.alpha[i] <= self.epi:
            #if -self.y[i]*self.e[i] - self.epi>0:
                #print 'kkt:',-self.y[i]*self.e[i] - self.epi
            return max(-self.y[i]*self.e[i] - self.epi, 0)
        if self.alpha[i] > 0. and self.alpha[i] < self.C:
            return np.abs(-self.y[i]*self.e[i] - self.epi)
        if self.alpha[i] >= self.C-self.epi:
            #if self.y[i]*self.e[i] - self.epi>0:
                #print 'kkt3:',self.y[i]*self.e[i] - self.epi
            return max(self.y[i]*self.e[i] - self.epi, 0)
            
    def chooseA2(self,a1):
        maxi = -1
        maxi_i = -1
        #TODO　set threshold
        for i in range(self.n):
            self.calcE(i)
            #use this doesn't work temp = np.abs((self.e[a1] - self.e[i]) / (self.k[a1][a1]+self.k[i][i]-2*self.k[a1][i]))
            temp = np.abs(self.e[a1] - self.e[i])
            if temp > maxi:
                maxi = temp
                maxi_i = i
        #print 'maxi,',maxi
        return maxi_i
    
    def updateAlpha(self,a1):
        a2 = self.chooseA2(a1)
        if self.y[a1] == self.y[a2]:
            l = max(0,self.alpha[a2]+self.alpha[a1]-self.C)
            h = min(self.C,self.alpha[a2]+self.alpha[a1])
        else:
            l = max(0,self.alpha[a2]-self.alpha[a1])
            h = min(self.C,self.alpha[a2]-self.alpha[a1]+self.C) 
        alpha2_old =  self.alpha[a2]
        alpha1_old =  self.alpha[a1]     
        self.alpha[a2] += self.y[a2] * (self.e[a1] - self.e[a2]) / (self.k[a1][a1]+self.k[a2][a2]-2*self.k[a1][a2])
        calc = self.alpha[a2]
        #print 'alpha2:',self.alpha[a2]
        self.alpha[a2] = min(self.alpha[a2],h)
        self.alpha[a2] = max(self.alpha[a2],l)
        self.alpha[a1] += self.y[a1]*self.y[a2]*(alpha2_old-self.alpha[a2])
        #print 'a1:',a1,' ', self.alpha[a1],' ',alpha1_old,'a2:',a2,' ',self.alpha[a2],' ',alpha2_old,' ',calc
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
    
    def print_kkt(self):
        print 'KKT:'
        temp = 0
        flag = True
        for i in range(self.n):
            temp += self.alpha[i]*self.y[i]
        print 'sum alpha*y:%f'%temp
        if temp > self.epi:
            flag = False
        num = 0
        for i in range(self.n):
            if self.alpha[i] > self.epi and self.alpha[i] < self.C - self.epi and self.e[i]*self.y[i]>self.epi:
                num += 1
                flag = False
                print '1alpha_',i,':',self.alpha[i],'y*e_i',self.e[i]*self.y[i]
            if self.alpha[i] <= self.epi and self.e[i]*self.y[i]<-self.epi:
                num += 1
                flag = False
                print '2alpha_',i,':',self.alpha[i],'y*e_i',self.e[i]*self.y[i]
            if self.alpha[i]>self.C-self.epi and self.e[i]*self.y[i]>-self.epi:
                num += 1
                flag = False
                print '3alpha_',i,':',self.alpha[i],'y*e_i',self.e[i]*self.y[i]
        print 'break num:',num
        return flag
                
        
    def train(self,x,y,iter = 100):
        self.n = x.shape[0]
        self.alpha = np.zeros((self.n,),dtype = float)
        self.e = np.zeros((self.n,),dtype = float)
        self.x = x
        self.y = y
        self.uptimes = np.zeros((self.n,),dtype = int)
        self.total_time = 0
        self.k = np.zeros((self.n,self.n),dtype = float)
        for i in range(self.n):
            for j in range(i,self.n):
                self.k[i][j] = self.kernel(x[i],x[j])
                self.k[j][i] = self.k[i][j]
        #print self.k
        for i in range(self.n):
            self.calcE(i)
        old = -1
        test_p = iter / 10
        for i in range(iter):
            if i % test_p == 0:
                if(self.print_kkt()):
                    break
            maxi_list = []
            print 'iter:',i
            flag = False
            maxi = -1
            ran = np.nonzero(self.alpha)[0]
            for j in ran:
                self.calcE(j)
                if self.alpha[j] > self.C-self.epi:
                    continue
                temp = self.KKT(j)
                if temp > maxi + self.epi and self.uptimes[j]<=1.3*(self.total_time)/self.n:
                    maxi_list = []
                    maxi_list.append(j)
                    maxi = temp
                elif np.abs(temp - maxi) < self.epi and self.uptimes[j]<=1.3*(self.total_time)/self.n:
                    maxi_list.append(j)
            if maxi > 0.01:
                #print 'maxi_list:',maxi_list
                self.total_time += 1
                tmp = np.random.randint(0,len(maxi_list))
                self.uptimes[maxi_list[tmp]] += 1
                self.updateAlpha(maxi_list[tmp])
                flag = True
            maxi = -1
            if not flag:
                maxi_list = []
                #print 'not flag'
                for j in range(self.n):
                    self.calcE(j)
                    temp = self.KKT(j) 
                    if temp > maxi + self.epi and self.uptimes[j]<=1.3*(self.total_time)/self.n:
                        maxi = temp
                        maxi_list = []
                        maxi_list.append(j)
                        #print 'temp',temp
                    elif np.abs(temp - maxi) < self.epi and self.uptimes[j]<=1.3*(self.total_time)/self.n:
                        maxi_list.append(j)
                #print 'maxi:',maxi,'maxi_i:',maxi_i,' ',self.KKT(maxi_i)
                #print 'maxi_list:',maxi_list
                self.total_time += 1
                tmp = np.random.randint(0,len(maxi_list))
                self.uptimes[maxi_list[tmp]] += 1
                self.updateAlpha(maxi_list[tmp]) 
        self.sv = []
        for i in range(self.n):
            if self.alpha[i] > 0:
                self.sv.append(i)
                plt.plot(self.x[i][0],self.x[i][1],'oy')
        self.print_kkt()
    def predict(self,x):
        out = np.zeros(x.shape[0])
        #print x
        for j in range(x.shape[0]):
            for i in self.sv:
                out[j] += self.alpha[i]*self.y[i]*self.kernel(self.x[i],x[j])
        out += self.b
        #print out
        #print self.alpha
        #print 'e',self.e
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

def Polynomial(x,z,p=3):  
    return np.sum((x*z + 1)**p)
        
if __name__ == '__main__':
    
    X,y = datasets.make_moons(100,noise=0.01)
    #X,y = datasets.make_circles(100,noise=0.01)
    for i in range(len(y)):
        if y[i] == 0:
            y[i] = -1
    svm = svm(10, Polynomial,0.01)
    svm.train(X,y,1000)
    print 'error rate:', svm.error(X,y)
    xx, yy = np.meshgrid(np.arange(X[:,0].min()-0.3, X[:,0].max()+0.3, 0.3),
                     np.arange(X[:,1].min()-0.3, X[:,0].max()+0.3, 0.3))
    Z = []
    x_t = np.c_[xx.ravel(), yy.ravel()]  
    Z = svm.predict(x_t)
    Z = np.array(Z).reshape(xx.shape)
    cm = plt.cm.RdBu
    plt.contourf(xx, yy, Z, cmap=cm, alpha=.2)
    plt.scatter(X[:,0],X[:,1], s=75, c=svm.predict(X), alpha=.5)  
    plt.show()   
    print svm.uptimes