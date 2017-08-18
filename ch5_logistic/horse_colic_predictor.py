#!/usr/bin/python
# -*- coding: utf-8 -*-

import numpy
from numpy import *

def get_training_data(file_name):
    data_mat = []
    label_mat = []
    fd = open(file_name, 'r')
    for line in fd.readlines():
        line = line.strip()
        separate_vec = line.split('\t')
        for i in range(len(separate_vec)):
        	separate_vec[i] = float(separate_vec[i])
        label_mat.append(separate_vec[-1])
        del(separate_vec[-1])
        data_mat.append(separate_vec)
    
    fd.close()    
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
    
def logistic_regression(data_mat, label_mat):
	alpha = 0.01
	m, n = numpy.shape(data_mat)
	weights = numpy.ones(n)
	for i in range(m):
		z = sum(data_mat[i] * weights)
		h = sigmoid(z)
		weights = weights + alpha * (label_mat[i] - h) * numpy.array(data_mat[i])
	return weights

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

def get_test_data(file_name):
    data_mat = []
    label_mat = []
    fd = open(file_name, 'r')
    for line in fd.readlines():
        line = line.strip()
        separate_vec = line.split('\t')
        for i in range(len(separate_vec)):
        	separate_vec[i] = float(separate_vec[i])
        label_mat.append(separate_vec[-1])
        del(separate_vec[-1])
        data_mat.append(separate_vec)
    
    fd.close()    
    return data_mat, label_mat

def logistic_prediction(test_data_mat, test_label_mat, weights):
	determin = 0
	right = 0
	test_data_matrix = numpy.mat(test_data_mat)
	m, n = numpy.shape(test_data_matrix)
	total = m
	for i in range(m):
		z = sum(test_data_mat[i] * weights)
		h = sigmoid(z)
		if int(h) > 0.5:
			determin = 1
			if int(test_label_mat[i]) == determin:
				right += 1
		else:
			determin = 0
			if int(test_label_mat[i]) == determin:
				right += 1
	right_percentage = float(right)/total
	return right_percentage


if __name__ == "__main__":
    training_data_mat, training_label_mat = get_training_data('./horseColicTraining.txt')
    #weights = grad_ascent(training_data_mat, training_label_mat)
    weights = upgrade_random_grad_ascent(training_data_mat, training_label_mat)
    test_data_mat, test_label_mat = get_test_data('./horseColicTest.txt')
    percentage = logistic_prediction(test_data_mat, test_label_mat, weights)
    print percentage
