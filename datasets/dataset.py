#coding=utf-8
import numpy as np


def classification_dataset_produce(func, input_interval, num):
    '''
    func: real data discriminant function
    input_interval: the interval of every dimension of input data
    num: the number of datas to produce
    '''
    if not isinstance(input_interval, np.ndarray):
        input_interval = np.array(input_interval)
    
    input_interval = input_interval.astype(float)
    
    average = input_interval.mean(1)
    span = input_interval.transpose()[1]-input_interval.transpose()[0]
    print average,span
    
    dimension = len(input_interval)

    data = np.ndarray((dimension,num))
    
    for i in range(dimension):
        data[i] = average[i]
        data[i] += span[i]*np.random.random(num) - span[i] / 2
    
    data = data.transpose()
    
    