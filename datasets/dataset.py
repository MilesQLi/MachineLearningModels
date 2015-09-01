#coding=utf-8
import numpy as np
import matplotlib.pyplot as plt


def classification_dataset_produce(func, input_interval, num):
    '''
    func: real data discriminant function
    input_interval: the interval of every dimension of input data
    num: the number of datas to produce
    
    return the data set in the form [[input,output]] which input is array output is number
    '''
    if not isinstance(input_interval, np.ndarray):
        input_interval = np.array(input_interval)
    
    input_interval = input_interval.astype(float)
    
    average = input_interval.mean(1)
    span = input_interval.transpose()[1]-input_interval.transpose()[0]
    
    dimension = len(input_interval)

    datas = np.ndarray((dimension,num))
    
    for i in range(dimension):
        datas[i] = average[i]
        datas[i] += span[i]*np.random.random(num) - span[i] / 2
    
    datas = datas.transpose()
    
    outputs = []
    
    for data in datas:
        outputs.append(func(data))
        
    return [(data,output) for (data,output) in zip(datas,outputs)]
    '''    
    outputs = np.array(outputs)
    outputs.shape=(outputs.shape[0],1)
    return np.hstack((datas,outputs))
    '''

def plot_data(x,y):
    '''
    x must be 2 dimension
    '''
    a = 5
    plt.plot(x, y)
    plt.show()