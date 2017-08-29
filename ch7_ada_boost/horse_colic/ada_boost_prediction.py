#!/usr/bin/python
# -*- coding: utf-8 -*-

import numpy

def load_data(file_name):
    fd = open(file_name)
    feature_num = len(fd.readline().split('\t')) - 1
    data_mat = []
    label_mat = []
    for line in fd.readlines():
        line_feature = []
        one_line = line.strip()
        feature_arr = one_line.split('\t')
        for i in range(feature_num):
            line_feature.append(float(feature_arr[i]))
        data_mat.append(line_feature)
        label_mat.append(float(feature_arr[-1]))
    fd.close()
    return data_mat, label_mat

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
        #print "D:", D.T
        alpha = float(0.5 * numpy.log((1.0 - error)/numpy.max(error, 1e-16)))
        best_stump['alpha'] = alpha
        weak_class_array.append(best_stump)
        #print "class est:", class_est.T
        expon = numpy.multiply(-1 * alpha * numpy.mat(class_labels).T, class_est)
        D = numpy.multiply(D, numpy.exp(expon))
        D = D/D.sum()
        agg_class_est += alpha * class_est
        #print "agg_class_est:" , agg_class_est.T
        agg_errors = numpy.multiply(numpy.sign(agg_class_est) != numpy.mat(class_labels).T, numpy.ones((m, 1)))
        err_rate = agg_errors.sum()/m
        #print "total error:", err_rate, "\n"
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
    #change this number will change error rate
    classifier_num = 50
    training_data_mat, training_class_label = load_data('./horseColicTraining2.txt')
    classifier_arr = ada_boost_trains_ds(training_data_mat, training_class_label, classifier_num)
    #print "classifier: ", classifier_arr
    #print "predict result: ", ada_classify([[5, 5], [0, 0]], classifier_arr)
    testing_data_mat, testing_class_label = load_data('./horseColicTest2.txt')
    predict_result = ada_classify(testing_data_mat, classifier_arr)
    total_num = len(testing_data_mat)
    err_arr = numpy.mat(numpy.ones((total_num, 1)))
    err_num = err_arr[predict_result != numpy.mat(testing_class_label).T].sum()
    err_rate = float(err_num) / total_num
    print "error rate: ", err_rate * 100, "%"



