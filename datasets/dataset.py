# coding=utf-8
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
    span = input_interval.transpose()[1] - input_interval.transpose()[0]
    
    dimension = len(input_interval)

    datas = np.ndarray((dimension, num))
    
    for i in range(dimension):
        datas[i] = average[i]
        datas[i] += span[i] * np.random.random(num) - span[i] / 2
    
    datas = datas.transpose()
    
    outputs = []
    
    for data in datas:
        outputs.append(func(data))
        
    return [(data, output) for (data, output) in zip(datas, outputs)]
    '''    
    outputs = np.array(outputs)
    outputs.shape=(outputs.shape[0],1)
    return np.hstack((datas,outputs))
    '''

def plot_data(x, y, input_interval, contour = None):
    '''
    x must be 2 dimension
    '''
    colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k', 'w']
    max = np.array(y).max()
    classes = []
    for i in range(max + 1):
        cla = []
        for j in range(len(x)):
            if y[j] == i:
                cla.append(x[j])
        classes.append(cla)
    
    for i in range(len(classes)):
        print classes[i]
        plt.plot([x[0] for x in classes[i]], [x[1] for x in classes[i]], colors[i] + 'o')
    plt.xlim(input_interval[0][0] - (input_interval[0][1] - input_interval[0][0]) / 10., input_interval[0][1] + (input_interval[0][1] - input_interval[0][0]) / 10.)
    plt.ylim(input_interval[1][0] - (input_interval[1][1] - input_interval[1][0]) / 10., input_interval[1][1] + (input_interval[1][1] - input_interval[1][0]) / 10.)
    if contour is not None:
        cm = plt.cm.RdBu
        plt.contourf(contour[0], contour[1], contour[2], cmap=cm, alpha=.2)
    plt.show()
