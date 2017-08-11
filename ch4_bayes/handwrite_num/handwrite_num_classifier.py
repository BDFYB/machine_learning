import numpy
import re
import os
import random


def train_nb0(train_matrix, train_category):
    #p(ci|xi) = p(xi|ci) * p(ci) / p(xi)
    
    #calculate p(ci), i = 0~9
    category_list = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    for category in train_category:
        category_list[category] += 1
    category_possiblity = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    for i in range(10):
        category_possiblity[i] = float(category_list[i]) / sum(category_list)
  
    #calculate p(xi|ci)
    p_num = numpy.ones((10, len(train_matrix[0])))
    p_denom = numpy.ones(10) * 2
    num_train_docs = len(train_matrix)
    for i in range(num_train_docs):
        p_num[train_category[i]] += train_matrix[i]
        p_denom[train_category[i]] += sum(train_matrix[i])
        #p_denom[train_category[i]] += 1

    for i in range(10):
        for j in range(len(p_num[i])):
            p_num[i][j] = numpy.log(p_num[i][j] / p_denom[i])

    return p_num, category_possiblity

def classify_using_nb(target_vec, p_vec, category_possiblity):
    possible_list = []
    max_possible = -1000000
    max_possition = 0
    for i in range(len(p_vec)):
        cur_possible = sum(target_vec * p_vec[i]) + numpy.log(category_possiblity[i])
        possible_list.append(cur_possible)
        if cur_possible > max_possible:
            max_possible = cur_possible
            max_possition = i
    return max_possition, possible_list


if __name__ == "__main__":
    trainning_files = os.listdir("./trainingDigits")
    #training
    training_mat = []
    training_label = []
    for one_file in trainning_files:
        single_vector = []
        fd = open("./trainingDigits/%s" % one_file)
        training_label.append(int(one_file.split('_')[0]))
        for i in range(32):
            line = fd.readline()
            for index in range(32):
                single_vector.append(int(line[index]))
        training_mat.append(single_vector)

    p_vec, category_possiblity = train_nb0(training_mat, training_label)
    #print category_possiblity
    #print p_vec

    #testing
    testing_file = os.listdir("./testDigits")
    err_num = 0
    total_num = len(testing_file)
    for one_file in testing_file:
        actual = int(one_file.split('_')[0])
        fd = open("./testDigits/%s" % one_file)
        test_vec = []
        for i in range(32):
            line = fd.readline()
            for index in range(32):
                test_vec.append(int(line[index])) 
        determine, possible_list = classify_using_nb(test_vec, p_vec, category_possiblity)   
        if determine != actual:
            err_num += 1
            print "actual %s determined as %s, file: %s " % (actual, determine, one_file) 
            print "possible: %s: %s and %s: %s" % (actual, possible_list[actual], determine, possible_list[determine])  
            print possible_list
    failed_rate = err_num / float(total_num)
    print "failed percentage: %s" % failed_rate















