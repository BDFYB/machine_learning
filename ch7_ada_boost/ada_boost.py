#!/usr/bin/python
# -*- coding: utf-8 -*-

import numpy

def load_simple_data():
    dat_mat = numpy.matrix([[1., 2.1], 
                           [2., 1.1],
                           [1.3, 1.],
                           [1. , 1.],
                           [2. , 1.]])
    class_labels = [1.0, 1.0, -1.0, -1.0, 1.0]
    return dat_mat, class_labels

def stump_classify(data_mat, dimen, thresh_val, thres_ineq):
    #单层决策树，通过数组过滤实现
    ret_array = numpy.ones((numpy.shape(data_mat)[0], 1))
    if thres_ineq == 'lt':
        ret_array[data_mat[:, dimen] <= thresh_val] = -1.0
    else:
        ret_array[data_mat[:, dimen] > thresh_val] = -1.0
    return ret_array

def build_stump(data_array, class_labels, D):
    #找到错误率最小的单层决策树
    data_matrix = numpy.mat(data_array)
    label_matrix = numpy.mat(class_labels).T
    m, n = numpy.shape(data_matrix)
    num_steps = 10.0
    best_stump = {}
    best_clas_est = numpy.mat(numpy.zeros((m,1)))
    min_error = numpy.inf
    for i in range(n):
        range_min = data_matrix[:, i].min()
        range_max = data_matrix[:, i].max()
        step_size = (range_max - range_min) / num_steps
        for j in range(-1, int(num_steps) + 1):
            for inequal in ['lt', 'gt']:
                thresh_val = (range_min + float(j) * step_size)
                predicted = stump_classify(data_matrix,i,thresh_val,inequal)
                err_arr = numpy.mat(numpy.ones((m, 1)))
                err_arr[predicted == label_matrix] = 0
                weighted_error = D.T * err_arr
                #print "split: dim %d, thresh %.2f, thresh ineqal: %s, the weighted error is %.3f" % (i, thresh_val, inequal, weighted_error)
                if weighted_error < min_error:
                    min_error = weighted_error
                    best_clas_est = predicted.copy()
                    best_stump['dim'] = i
                    best_stump['thresh'] = thresh_val
                    best_stump['ineq'] = inequal
    return best_stump, min_error, best_clas_est

def ada_boost_trains_ds(data_array, class_labels, num_it = 40):
    weak_class_array = []
    m = numpy.shape(data_array)[0]
    D = numpy.mat(numpy.ones((m, 1))/m)
    agg_class_est = numpy.mat(numpy.zeros((m, 1)))
    for i in range(num_it):
        best_stump, error, class_est = build_stump(data_array, class_labels, D)
        print "D:", D.T
        alpha = float(0.5 * numpy.log((1.0 - error)/numpy.max(error, 1e-16)))
        best_stump['alpha'] = alpha
        weak_class_array.append(best_stump)
        print "class est:", class_est.T
        expon = numpy.multiply(-1 * alpha * numpy.mat(class_labels).T, class_est)
        D = numpy.multiply(D, numpy.exp(expon))
        D = D/D.sum()
        agg_class_est += alpha * class_est
        print "agg_class_est:" , agg_class_est.T
        agg_errors = numpy.multiply(numpy.sign(agg_class_est) != numpy.mat(class_labels).T, numpy.ones((m, 1)))
        err_rate = agg_errors.sum()/m
        print "total error:", err_rate, "\n"
        if err_rate == 0.0:
            break
    return weak_class_array

def ada_classify(dat_to_class, classifier_arr):
    data_mat = numpy.mat(dat_to_class)
    m = numpy.shape(data_mat)[0]
    agg_class_est = numpy.mat(numpy.zeros((m, 1)))
    for i in range(len(classifier_arr)):
        class_est = stump_classify(data_mat, classifier_arr[i]['dim'],\
                                   classifier_arr[i]['thresh'],\
                                   classifier_arr[i]['ineq'])
        agg_class_est += classifier_arr[i]['alpha'] * class_est
        #print agg_class_est
    return numpy.sign(agg_class_est)

if __name__ == "__main__":
    data_mat, class_label = load_simple_data()
    #D = numpy.mat(numpy.ones((5, 1)) / 5)
    #best_stump, min_error, best_clas_est = build_stump(data_mat, class_label, D)
    classifier_arr = ada_boost_trains_ds(data_mat, class_label, 9)
    print "classifier: ", classifier_arr
    print "predict result: ", ada_classify([[5, 5], [0, 0]], classifier_arr)
    


