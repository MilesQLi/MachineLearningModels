#coding=utf-8
import numpy as np
from scipy.linalg import svd
from sklearn.datasets import fetch_olivetti_faces
from numpy.random import RandomState
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
import time
class PCA(object):
    def __init__(self,n_component):
        self.n_component = n_component
    
    def train(self,X):
        m,n = X.shape
        self.mean = np.average(X, axis = 1).reshape(X.shape[0],1)
        X_ = X - self.mean
        if m>n:
            U,S,V = svd(X_.transpose())
            self.U = U
        else:
            U,S,V = svd(X_)
            self.U = V.transpose()
            
    def transform(self,X):
        X_ = X - self.mean
        return np.dot(X_,self.U[:,:self.n_component])
    def reconstruct(self,X):
        X_ = X - self.mean
        temp1 = np.dot(X_,self.U[:,:self.n_component])
        temp2 = np.dot(temp1,self.U[:,:self.n_component].transpose())
        result = temp2 + self.mean   
        print 'mean error:',np.sum((X-result)**2)/(X.shape[0]*X.shape[1])
        return result

def plot_gallery(title, images, n_col=3, n_row=2):
    '''
    get from sklearn
    '''
    plt.figure(figsize=(2. * n_col, 2.26 * n_row))
    plt.suptitle(title, size=16)
    for i, comp in enumerate(images):
        plt.subplot(n_row, n_col, i + 1)
        vmax = max(comp.max(), -comp.min())
        plt.imshow(comp.reshape((64, 64)), cmap=plt.cm.gray,
                   interpolation='nearest',
                   vmin=-vmax, vmax=vmax)
        plt.xticks(())
        plt.yticks(())
    plt.subplots_adjust(0.01, 0.05, 0.99, 0.93, 0.04, 0.)
'''
#test success  
if __name__ == '__main__':
    iris = load_iris()
    X = iris.data
    y = iris.target
    
    n_components = 2
    
    pca = PCA(n_components)
    pca.train(X)
    X_pca = pca.transform(X)
    plt.figure(figsize=(8, 8))
    for c, i, target_name in zip("rgb", [0, 1, 2], iris.target_names):
            plt.scatter(X_pca[y == i, 0], X_pca[y == i, 1],
                        c=c, label=target_name)
    
    
    plt.title("PCA" + " of iris dataset")
    plt.legend(loc="best")
    plt.axis([-5, -1, -4.5, 4.5])
    
    plt.show()

'''
#test success    
if __name__ == '__main__':
    rng = RandomState(0)
    dataset = fetch_olivetti_faces(shuffle=True, random_state=rng)
    faces = dataset.data  
    plot_gallery('origin', faces[:6])
    tima  = 0
    for i in range(5,50,5):    
        print 'n_comonents:',i
        pca = PCA(i)
        t1 = time.clock()
        pca.train(faces)
        tima += time.clock()-t1
        recon = pca.reconstruct(faces)
        plot_gallery('recon %d components'%i, recon[:6])
    plt.show()
    print tima
