# -*- coding: utf-8 -*-
"""
Created on Sun Jul 21 21:49:26 2019

@author: WW
"""

"""
python: @property的使用
perceptron:
    1. 原始方法，便利数据，直到全部>0
    2. 对偶方法：在高维数据中，降低计算复杂度
"""

import numpy as np

class Perceptron:  # 感知机
    def __init__(self, dataSet, labels):  # 初始化数据集和标签, initial dataset and label
        self._dataSet = np.array(dataSet)
        self._labels = np.array(labels).transpose()
 
    def originTrain(self):
        m, n = np.shape(self.dataSet)
        weights, bias, flag = np.zeros([1, n]), 0, False
        while flag != True:
            flag = True
            for i in range(m):
                y = weights * np.mat(dataSet[i]).T + bias  
                if (self.sign(y) * self.labels[i] < 0):  
                    weights += self.labels[i] * self.dataSet[i]
                    bias += self.labels[i]  # 更新偏置
                    print("weights %s, bias %s" % (weights, bias))
                    flag = False
        return weights, bias
   
    #对偶方式 
    def dualTrain(self):
        m, n = np.shape(dataSet)
        
        #notice:要转化为数组，而非矩阵
        gram = np.array(dataSet * np.mat(dataSet).T)
        print(gram)
        
        a, weights, bias, flag = np.zeros(m), np.zeros([1,n]), 0, False
        while not flag :
            flag = True
            for i, xi in enumerate(dataSet):
                sum = 0
                for j, xj in enumerate(dataSet):
                    sum += a[j] * labels[j] * gram[j][i]
                sum += bias
                if sum * labels[i] <= 0:
                    a[i] += 1
                    bias += labels[i]
                    flag = False
            for i in range(m):
                weights += a[i] * self.dataSet[i] * self.labels[i]
        return weights, bias
        
        
        
    
 
    def sign(self, y):  # 符号函数 sign function
        if (y > 0):
            return 1
        else:
            return -1
 
    @property
    def dataSet(self):
        return self._dataSet
 
    @property
    def labels(self):
        return self._labels
 
if __name__ == "__main__":
    dataSet = [[3, 3],
               [4, 3],
               [1, 1]]
    labels = [1, 1, -1]
    perceptron = Perceptron(dataSet, labels)  # 创建一个感知机对象
    weights, bias = perceptron.originTrain()  # 训练
    print("final weights:%s, bias:%s" % (weights, bias))
    print("----- dual ------")
    weights, bias = perceptron.dualTrain()  # 训练
    print("final weights:%s, bias:%s" % (weights, bias))
    
    