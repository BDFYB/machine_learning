#!/usr/bin/python
# -*- coding: utf-8 -*-

import numpy
from numpy import *
import random

def load_data(file_name):
	fd = open(file_name)
	data_mat = []
	label_mat = []
	for line in fd.readlines():
		line = line.strip()
		data_arr = line.split('\t')
		data_mat.append([float(data_arr[0]), float(data_arr[1])])
		label_mat.append(float(data_arr[-1]))
	fd.close()
	return data_mat, label_mat

def get_random(i, m):
	j = i
	while j == i:
		j = int(random.uniform(0, m))
	return j

def clip_alpha(aj, h, l):
	if aj > h:
		aj = h
	if l > aj:
		aj = l
	return aj

def kernel_trans(x, y, type = 'line', sigma = 4):
    m, n = shape(x)
    K = numpy.mat(zeros([m, 1]))
    if type == 'line':
        K = x * y.T
    elif type == 'rbf':
        for i in range(m):
            derta = x[i,:] - y
            K[i] = numpy.exp(-derta*derta.T/(2*sigma*sigma))
    return K

def smo_with_kernel(dataMatIn, classLabels, C, toler, maxIter):
    dataMatrix = mat(dataMatIn)
    labelMat = mat(classLabels).transpose()
    b = 0; m,n = shape(dataMatrix)
    alphas = mat(zeros((m,1)))
    sigma = 2
    iter = 0
    while (iter < maxIter):
        alphaPairsChanged = 0
        for i in range(m):
            fXi = float(multiply(alphas,labelMat).T*kernel_trans(dataMatrix, dataMatrix[i,:], 'rbf', sigma )) + b
            Ei = fXi - float(labelMat[i])#if checks if an example violates KKT conditions
            if ((labelMat[i]*Ei < -toler) and (alphas[i] < C)) or ((labelMat[i]*Ei > toler) and (alphas[i] > 0)):
                j = get_random(i,m)
                fXj = float(multiply(alphas,labelMat).T*kernel_trans(dataMatrix, dataMatrix[j,:], 'rbf', sigma )) + b
                Ej = fXj - float(labelMat[j])
                alphaIold = alphas[i].copy()
                alphaJold = alphas[j].copy()
                if (labelMat[i] != labelMat[j]):
                    L = max(0, alphas[j] - alphas[i])
                    H = min(C, C + alphas[j] - alphas[i])
                else:
                    L = max(0, alphas[j] + alphas[i] - C)
                    H = min(C, alphas[j] + alphas[i])
                if L==H: 
                    #print "L==H"
                    continue
                eta = 2.0 * kernel_trans(dataMatrix[i,:], dataMatrix[j,:], 'rbf', sigma) - kernel_trans(dataMatrix[i,:], dataMatrix[i,:], 'rbf', sigma) - kernel_trans(dataMatrix[j,:], dataMatrix[j,:], 'rbf', sigma)
                if eta >= 0: 
                    #print "eta>=0"
                    continue
                alphas[j] -= labelMat[j]*(Ei - Ej)/eta
                alphas[j] = clip_alpha(alphas[j],H,L)
                if (abs(alphas[j] - alphaJold) < 0.00001): 
                    #print "j not moving enough"
                    continue
                alphas[i] += labelMat[j]*labelMat[i]*(alphaJold - alphas[j])#update i by the same amount as j
                                                                        #the update is in the oppostie direction
                b1 = b - Ei- labelMat[i]*(alphas[i]-alphaIold)*kernel_trans(dataMatrix[i,:], dataMatrix[i,:], 'rbf', sigma) - labelMat[j]*(alphas[j]-alphaJold)*kernel_trans(dataMatrix[j,:], dataMatrix[j,:], 'rbf', sigma)
                b2 = b - Ej- labelMat[i]*(alphas[i]-alphaIold)*kernel_trans(dataMatrix[i,:], dataMatrix[j,:], 'rbf', sigma) - labelMat[j]*(alphas[j]-alphaJold)*kernel_trans(dataMatrix[j,:], dataMatrix[j,:], 'rbf', sigma)
                if (0 < alphas[i]) and (C > alphas[i]): 
                	b = b1
                elif (0 < alphas[j]) and (C > alphas[j]): 
                	b = b2
                else: 
                	b = (b1 + b2)/2.0
                alphaPairsChanged += 1
                #print "iter: %d i:%d, pairs changed %d" % (iter,i,alphaPairsChanged)
        if (alphaPairsChanged == 0): iter += 1
        else: iter = 0
        #print "iteration number: %d" % iter
    return b,alphas

def calculate_w(alphas, data_arr, class_labels):
    x = numpy.mat(data_arr)
    label_mat = numpy.mat(class_labels).T
    m, n = shape(x)
    w = zeros([n, 1])
    for i in range(m):
        w += numpy.multiply(alphas[i] * label_mat[i], x[i].T)
    return w


if __name__ == "__main__":
    data_mat, label_mat = load_data("./testSetRBF.txt")
    b, alphas = smo_with_kernel(data_mat, label_mat, 0.6, 0.001, 40)
    support_vec = 0
    for i in range(len(alphas)):
        if alphas[i] != 0:
            support_vec += 1
            #print 'supprot vector:', data_mat[i], label_mat[i]
    print 'num of support vecotr: %d' % support_vec
    w = calculate_w(alphas, data_mat, label_mat)
    det = 0
    total_no = len(data_mat)
    err_no = 0
    tmp_w = sum(alphas * numpy.mat(label_mat))
    print tmp_w
    for i in range(total_no):
        total_no += 1
        #determine = w.T * numpy.mat(data_mat[i]).T + b
        determine = tmp_w * kernel_trans(numpy.mat(data_mat)[i,:], numpy.mat(data_mat)[i,:], 'rbf', 4) + b
        if determine > 0:
            det = 1
        else:
            det = -1
        if det != label_mat[i]:
            err_no += 1

    print float(err_no) / total_no












