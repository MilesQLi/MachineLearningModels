#coding=utf-8


'''
classification


iris:
def loadDataset(filename, split, trainingSet=[] , testSet=[]):
    with open(filename, 'rb') as csvfile:
        lines = csv.reader(csvfile)
        dataset = list(lines)
        for x in range(len(dataset)-1):
            for y in range(4):
                dataset[x][y] = float(dataset[x][y])
            if random.random() < split:
                trainingSet.append(dataset[x])
            else:
                testSet.append(dataset[x])
def main():
    # prepare data
    trainingSet=[]
    testSet=[]
    split = 0.67
    loadDataset('iris.data', split, trainingSet, testSet)
_______________________________    
from sklearn import datasets
if __name__ == '__main__':
    n_samples = 5000
    n_features = 20
    n_informative = 10
    n_redundant = 0
    n_classes = 2
    n_clusters_per_class = 1
    x,y = datasets.make_classification(n_samples = n_samples, n_redundant = n_redundant,n_informative=n_informative,n_clusters_per_class = n_clusters_per_class, n_features = n_features,n_classes = n_classes)

_______________________________ 



   
    
'''



'''
nlp:




'''

'''
image:

minist:
    f = gzip.open('mini_mnist.pkl.gz', 'rb')
    x = cPickle.load(f)
    y = cPickle.load(f)
_______________________________ 


'''



'''
regression:



'''







