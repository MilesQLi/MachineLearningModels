#coding=utf-8
import numpy as np
from scipy.linalg import svd
from sklearn.datasets import fetch_olivetti_faces
from numpy.random import RandomState
import matplotlib.pyplot as plt

class PCA(object):
    def __init__(self,n_component):
        self.n_component = n_component
    
    def train(self,X):
        m,n = X.shape
        self.mean = np.average(X, axis = 1).reshape(X.shape[0],1)
        X_ = X - self.mean
        self.U,self.S,self.V = svd(np.dot(X_.transpose(),X_))
    
    def transform(self,X):
        X_ = X - self.mean
        return np.dot(self.U[:,:self.n_component].transpose(),X_)
    def reconstruct(self,X):
        X_ = X - self.mean
        temp1 = np.dot(X_,self.U[:,:self.n_component])
        temp2 = np.dot(temp1,self.U[:,:self.n_component].transpose())
        return temp2 + self.mean   

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
    
if __name__ == '__main__':
    rng = RandomState(0)
    dataset = fetch_olivetti_faces(shuffle=True, random_state=rng)
    faces = dataset.data  
    plot_gallery('origin', faces[:6])
    for i in range(10,50,5):    
        print 'n_comonents:',i
        pca = PCA(50)
        pca.train(faces)
        recon = pca.reconstruct(faces)
        plot_gallery('recon %d components'%i, recon[:6])
    plt.show()