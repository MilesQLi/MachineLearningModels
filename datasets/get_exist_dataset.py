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
from sklearn import datasets
datasets.make_moons(sample number,shuffle or not,noise=0.21)

_______________________________
iris = datasets.load_iris()
X = iris.data
y = iris.target
    
'''



'''
nlp:

NER:

atis irline Travel Information System:
def atisfold(fold):
    filename = 'atis.fold'+str(fold)+'.pkl.gz'
    f = gzip.open(filename, 'rb')
    train_set, valid_set, test_set, dicts = cPickle.load(f)
    return train_set, valid_set, test_set, dicts

every set is a 3 tuple. the first item is the sentence with number represent the word, the third item is the labels to the sentences, the third item is what I don't know but maybe not useful here

dicts contains two dicts dicts['words2idx'] map word to index dicts['labels2idx'] maps label 2 index

__________



_____________________________________________
sentiment:





'''

'''
image:

mnist:
    f = gzip.open('mnist.pkl.gz', 'rb')
    train_set, valid_set, test_set = cPickle.load(f)
    train_x = train_set[0]
    train_y = train_set[1]

sklearn_digits:
digits = datasets.load_digits()
digits.data is (1797L, 64L)  data  digits.images is 1797 (8*8) array. digits.target is the corresponding classes
digits can also be used like dict
_______________________________ 




'''



'''
regression:



'''







