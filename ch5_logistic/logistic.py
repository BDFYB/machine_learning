#!/usr/bin/python
# -*- coding: utf-8 -*-
import numpy 
from numpy import *
import matplotlib.pyplot as plt

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

def random_grad_ascent(data_mat, label_mat):
    m, n = shape(data_mat)
    weights_total = []
    alpha = 0.01
    weights = ones(n)
    for i in range(m):
        weights_total.append(weights.tolist())
        h = sigmoid(sum(data_mat[i] * weights))
        error = label_mat[i] - h
        weights = weights + alpha * error * array(data_mat[i])
    #weights转为列向量
    return numpy.mat(weights).transpose(), weights_total

def upgrade_random_grad_ascent(data_mat, label_mat, num_iter = 150):
    m, n = shape(data_mat)
    weights = ones(n)
    for j in range(num_iter):
        data_index = range(m)
        for i in range(m):
            alpha = 4 / (1.0 + j + i) + 0.01
            rand_index = int(random.uniform(0, len(data_index)))
            h = sigmoid(sum(data_mat[rand_index] * weights))
            error = label_mat[rand_index] - h
            weights = weights + alpha * error * array(data_mat[rand_index])
            del(data_index[rand_index])
    return numpy.mat(weights).transpose()

def plot_separate_line(data_mat, label_mat, weights):
    data_arr = array(data_mat)
    point_num = shape(data_arr)[0]
    label1_x1 = []
    label1_x2 = []
    label0_x1 = []
    label0_x2 = []
    for i in range(point_num):
        if int(label_mat[i]) == 1:
            label1_x1.append(data_mat[i][1])
            label1_x2.append(data_mat[i][2])
        else:
            label0_x1.append(data_mat[i][1])
            label0_x2.append(data_mat[i][2])
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.scatter(label1_x1, label1_x2, s = 30, c = 'red', marker = 's')
    ax.scatter(label0_x1, label0_x2, s = 30, c = 'green')
    x = arange(-3.0, 3.0, 0.1)
    y = (-weights[0] - weights[1] * x)/weights[2]
    ylist = y.tolist()
    ax.plot(x, ylist[0])
    plt.xlabel('X1')
    plt.ylabel('X2')
    plt.show()

def plot_weights_total(random_weights_total):
    size = len(random_weights_total)
    x = []
    y = []
    for i in range(size):
        x.append(i)
        y.append(random_weights_total[i][2])

    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.scatter(x, y, s = 30, c = 'red', marker = 's')
    """
    x = arange(-3.0, 3.0, 0.1)
    y = (-weights[0] - weights[1] * x)/weights[2]
    ylist = y.tolist()
    ax.plot(x, ylist[0])
    """
    plt.xlabel('X1')
    plt.ylabel('X2')
    plt.show()

if __name__ == "__main__":
    data_mat, label_mat = load_dataset()
    weights = grad_ascent(data_mat, label_mat)
    print weights
    #plot_separate_line(data_mat, label_mat, weights)
    random_method_weights, random_weights_total = random_grad_ascent(data_mat, label_mat)
    print random_method_weights
    #plot_separate_line(data_mat, label_mat, random_method_weights)
    #绘制各参数的变化图
    #plot_weights_total(random_weights_total)
    upgrade_random_method_weights = upgrade_random_grad_ascent(data_mat, label_mat)
    plot_separate_line(data_mat, label_mat, upgrade_random_method_weights)


