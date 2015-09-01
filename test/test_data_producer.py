import sys
sys.path.append('../datasets/')
from dataset import *

if __name__ == '__main__':
    a=5
    classification_dataset_produce(a,[[5,6],[18,22]],10)