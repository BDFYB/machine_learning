# -*- coding: utf-8 -*-
import numpy 
from numpy import *

def load_dataset():
    data_mat = []
    label_mat = []
    fd = open('test_set.txt')
    for line in fd.readlines():
        line_sp = line.strip().split()
        data_mat.append([1.0, float(line_sp[0]), float(line_sp[1])])
        label_mat.append(int(line_sp[2]))
    return data_mat, label_mat

def sigmoid(z):
    return 1.0 / (1 + numpy.exp(-z))

def grad_ascent(data_mat, class_label):
    alpha = 0.001
    max_cycle = 500
    data_matrix = numpy.mat(data_mat)
    class_label_matrix = numpy.mat(class_label).transpose()
    m, n = numpy.shape(data_matrix)
    weights = numpy.ones((n, 1))
    for k in range(max_cycle):
        h = sigmoid(data_matrix * weights)
        error = (class_label_matrix - h)
        #transpose:矩阵转置
        weights = weights + alpha * data_matrix.transpose() * error
    return weights

if __name__ == "__main__":
    data_mat, label_mat = load_dataset()
    weights = grad_ascent(data_mat, label_mat)
    print weights

