# coding=utf-8
import numpy as np
from sklearn import datasets
import matplotlib.pyplot as plt
import gzip
import cPickle
import time
class svm(object):
    def __init__(self, x, y, C, kernel, epi=0.01):
        self.C = C
        self.epi = epi
        self.kernel = kernel
        self.b = 0
        self.n = x.shape[0]
        self.alpha = np.zeros((self.n,), dtype=float)
        self.e = np.zeros((self.n,), dtype=float)
        self.x = x
        self.y = y
        self.uptimes = np.zeros((self.n,), dtype=int)
        self.total_time = 0
        try:
            self.k = np.zeros((self.n, self.n), dtype=float)
            self.cached = True
            for i in range(self.n):
                for j in range(i, self.n):
                    self.k[i][j] = self.kernel(x[i], x[j])
                    self.k[j][i] = self.k[i][j]
        except BaseException:
            self.cached = False
            
        # print self.k
        '''
        for i in range(self.n):
            self.calcE(i)
        '''
    def calcE(self, i):
        temp = 0
        for j in range(self.n):
            if self.cached:
                temp += self.alpha[j] * self.y[j] * self.k[i][j]
            else:
                temp += self.alpha[j] * self.y[j] * self.kernel(i, j)
        self.e[i] = temp + self.b - self.y[i]
        # print 'ei',self.e[i]
            
            
    def chooseA2(self, a1):
        maxi = -1
        maxi_i = -1
        # TODO　set threshold
        for i in range(self.n):
            self.calcE(i)
            # use this doesn't work temp = np.abs((self.e[a1] - self.e[i]) / (self.k[a1][a1]+self.k[i][i]-2*self.k[a1][i]))
            temp = np.abs(self.e[a1] - self.e[i])
            if temp > maxi:
                maxi = temp
                maxi_i = i
        # print 'maxi,',maxi
        return maxi_i
    
    def updateAlpha(self, a1):
        a2 = self.chooseA2(a1)
        if self.y[a1] == self.y[a2]:
            l = max(0, self.alpha[a2] + self.alpha[a1] - self.C)
            h = min(self.C, self.alpha[a2] + self.alpha[a1])
        else:
            l = max(0, self.alpha[a2] - self.alpha[a1])
            h = min(self.C, self.alpha[a2] - self.alpha[a1] + self.C) 
        alpha2_old = self.alpha[a2]
        alpha1_old = self.alpha[a1]     
        if self.cached:
            self.alpha[a2] += self.y[a2] * (self.e[a1] - self.e[a2]) / (self.k[a1][a1] + self.k[a2][a2] - 2 * self.k[a1][a2])
        else:
            self.alpha[a2] += self.y[a2] * (self.e[a1] - self.e[a2]) / (self.kernel(a1, a1) + self.self.kernel(a2, a2) - 2 * self.self.kernel(a1, a2))
        calc = self.alpha[a2]
        # print 'alpha2:',self.alpha[a2]
        self.alpha[a2] = min(self.alpha[a2], h)
        self.alpha[a2] = max(self.alpha[a2], l)
        self.alpha[a1] += self.y[a1] * self.y[a2] * (alpha2_old - self.alpha[a2])
        # print 'a1:',a1,' ', self.alpha[a1],' ',alpha1_old,'a2:',a2,' ',self.alpha[a2],' ',alpha2_old,' ',calc
        if self.cached:
            b1 = -self.e[a1] - self.y[a1] * self.k[a1][a1] * (self.alpha[a1] - alpha1_old) - self.y[a2] * self.k[a2][a1] * (self.alpha[a2] - alpha2_old) + self.b
            b2 = -self.e[a2] - self.y[a1] * self.k[a1][a2] * (self.alpha[a1] - alpha1_old) - self.y[a2] * self.k[a2][a2] * (self.alpha[a2] - alpha2_old) + self.b
        else:
            b1 = -self.e[a1] - self.y[a1] * self.kernel(a1, a1) * (self.alpha[a1] - alpha1_old) - self.y[a2] * self.kernel(a2, a1) * (self.alpha[a2] - alpha2_old) + self.b
            b2 = -self.e[a2] - self.y[a1] * self.kernel(a1, a2) * (self.alpha[a1] - alpha1_old) - self.y[a2] * self.kernel(a2, a2) * (self.alpha[a2] - alpha2_old) + self.b
            
        if self.alpha[a1] > 0 and self.alpha[a1] < self.C:
            self.b = b1
        elif self.alpha[a2] > 0 and self.alpha[a2] < self.C:
            self.b = b2
        else:
            self.b = (b1 + b2) / 2
        self.calcE(a1)
        self.calcE(a2)
        return np.abs(self.alpha[a2] - alpha2_old)
    
    def print_kkt(self):
        print 'KKT:'
        temp = 0
        flag = True
        for i in range(self.n):
            temp += self.alpha[i] * self.y[i]
        print 'sum alpha*y:%f' % temp
        if temp > self.epi:
            flag = False
        num = 0
        for i in range(self.n):
            if self.alpha[i] > self.epi and self.alpha[i] < self.C - self.epi and self.e[i] * self.y[i] > self.epi:
                num += 1
                flag = False
                print '1alpha_', i, ':', self.alpha[i], 'y*e_i', self.e[i] * self.y[i]
            if self.alpha[i] <= self.epi and self.e[i] * self.y[i] < -self.epi:
                num += 1
                flag = False
                print '2alpha_', i, ':', self.alpha[i], 'y*e_i', self.e[i] * self.y[i]
            if self.alpha[i] > self.C - self.epi and self.e[i] * self.y[i] > -self.epi:
                num += 1
                flag = False
                print '3alpha_', i, ':', self.alpha[i], 'y*e_i', self.e[i] * self.y[i]
        print 'break num:', num
        return flag
    def KKT(self, i):
        if ((self.y[i] * self.e[i] < -self.epi) and (self.alpha[i] < self.C)) or \
        (((self.y[i] * self.e[i] > self.epi)) and (self.alpha[i] > 0)):
            return False
        return True    
    def train(self, iter=100):
        old = -1
        test_p = iter / 10
        flag = True
        for i in range(1, iter + 1):
            if flag == False:
                break
            flag = False
            # if i % test_p == 0:
            #    if(self.print_kkt()):
            #        break
            maxi_list = []
            print 'iter:', i, ' '
            flag = False
            maxi = -1
            ran = np.nonzero(self.alpha)[0]
            np.random.shuffle(ran)
            # print 'i:'
            for j in ran:
                self.calcE(j)
                if self.alpha[j] > self.C - self.epi:
                    continue
                if not self.KKT(j):
                    temp = self.updateAlpha(j)
                    # print j,' ',temp
                    if  temp > self.epi:
                        flag = True
                        print j
                        break
            if not flag:
                seq = range(self.n)
                np.random.shuffle(seq)
                for j in seq:
                    self.calcE(j)
                    if not self.KKT(j):
                        # print j,' ',
                        temp = self.updateAlpha(j)
                        # print j,' ',temp
                        if  temp > self.epi:
                            flag = True
                            print j
                            break
        self.sv = []
        for i in range(self.n):
            if self.alpha[i] > 0:
                self.sv.append(i)
                plt.plot(self.x[i][0], self.x[i][1], 'oy')
        # self.print_kkt()
                
    '''    
    #bad method for long time and low accuracy
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
    def train(self,iter = 100):
        old = -1
        test_p = iter / 10
        for i in range(1,iter+1):
            #if i % test_p == 0:
            #    if(self.print_kkt()):
            #        break
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
        #self.print_kkt()
    '''
    
    def predict(self, x):
        out = np.zeros(x.shape[0])
        # print x
        for j in range(x.shape[0]):
            for i in self.sv:
                out[j] += self.alpha[i] * self.y[i] * self.kernel(self.x[i], x[j])
        out += self.b
        # print out
        # print self.alpha
        # print 'e',self.e
        # print self.y
        out2 = np.zeros(x.shape[0])
        out2[out > 0] = 1
        out2[out <= 0] = -1
        return out2
    def error(self, x, y):
        a = 0.
        a += x.shape[0]
        return np.sum(self.predict(x) != y) / a
    
def Gauss_kernel(x, z, sigma=1):
        return np.exp(-np.sum((x - z) ** 2) / (2 * sigma ** 2))
    
def linear(x, z):  
    return np.sum(x * z)

def Polynomial(x, z, p=3):  
    return np.sum((x * z + 1) ** p)
'''
if __name__ == '__main__':
    x,y = datasets.make_classification(n_samples = 1000, n_redundant = 0,n_informative=6,n_clusters_per_class = 1, n_features = 10,n_classes = 2)
    y[y==0] = -1
    train_x = x[:-100]
    train_y = y[:-100]
    valid_x = x[-100:]
    valid_y = y[-100:]
    test_x = x[-100:]
    test_y = y[-100:]
    svm = svm(train_x,train_y,10, Gauss_kernel,0.1)
    error = 1.
    svm.train(10)
    error = svm.error(valid_x,valid_y)
    for i in range(10):
        svm.train(10)
        temp = svm.error(valid_x,valid_y)
        print 'i:',i,' valid error:',temp
    print 'test error:',svm.error(test_x, test_y),svm.error(train_x, train_y)
'''

'''
if __name__ == '__main__':
    digits = datasets.load_digits()
    train_x = digits.data[:-200]
    train_y = digits.target[:-200]
    valid_x = digits.data[-200:-100]
    valid_y = digits.target[-200:-100]
    test_x = digits.data[-100:]
    test_y = digits.target[-100:]
    svm = svm(train_x,train_y,10, Gauss_kernel,0.001)
    error = 1.
    svm.train(10)
    error = svm.error(valid_x,valid_y)
    for i in range(100):
        svm.train(10)
        temp = svm.error(valid_x,valid_y)
        print 'i:',i,' valid error:',temp
        if temp > error:
            break
        else:
            error = temp
    print 'test error:',svm.error(test_x, test_y)
'''
'''
if __name__ == '__main__':
    f = gzip.open('mnist.pkl.gz', 'rb')
    train_set, valid_set, test_set = cPickle.load(f)
    train_x = train_set[0]
    train_y = train_set[1]
    valid_x = valid_set[0]
    valid_y = valid_set[1]
    test_x = test_set[0]
    test_y = test_set[1]
    svm = svm(train_x,train_y,10, Gauss_kernel,0.001)
    error = 1.
    svm.train(100)
    error = svm.error(valid_x,valid_y)
    for i in range(100):
        svm.train(10)
        temp = svm.error(valid_x,valid_y)
        print 'i:',i,' valid error:',temp
        if temp > error:
            break
        else:
            error = temp
    print 'test error:',svm.error(test_x, test_y)
'''
  
if __name__ == '__main__':
    
    X, y = datasets.make_moons(150, noise=0.01)
    # X,y = datasets.make_circles(100,noise=0.01)
    for i in range(len(y)):
        if y[i] == 0:
            y[i] = -1
    svm = svm(X[:100], y[:100], 10, Gauss_kernel, 0.0000001)
    t1 = time.clock()
    svm.train(1000)
    print 'train time:', time.clock() - t1
    print 'error rate:', svm.error(X, y)
    xx, yy = np.meshgrid(np.arange(X[:, 0].min() - 0.3, X[:, 0].max() + 0.3, 0.3),
                     np.arange(X[:, 1].min() - 0.3, X[:, 0].max() + 0.3, 0.3))
    Z = []
    x_t = np.c_[xx.ravel(), yy.ravel()]  
    Z = svm.predict(x_t)
    Z = np.array(Z).reshape(xx.shape)
    cm = plt.cm.RdBu
    plt.contourf(xx, yy, Z, cmap=cm, alpha=.2)
    plt.scatter(X[:, 0], X[:, 1], s=75, c=svm.predict(X), alpha=.5)  
    plt.show()   
    print svm.error(X[-50:], y[-50:])
