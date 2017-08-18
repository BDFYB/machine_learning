#!/usr/bin/python
# -*- coding: utf-8 -*-

import numpy

#测试matrix一般用途
def test_matrix():
    print "\tstart matrix test"
    mat1 = numpy.mat('1,1;3,2')
    mat2 = numpy.mat('1,2;2,1')
    mat3 = numpy.mat('1,2,2;1,3,2')
    print "mat1:\n %s" % mat1
    print "mat2:\n %s" % mat2
    print "mat1T:\n %s" % mat1.T
    print "|mat1|:\n %s"% numpy.linalg.det(mat1)
    print "mat1I:\n %s" % mat1.I
    #两个矩阵*是矩阵相乘，dot也一样为矩阵相乘
    print "mat1 * mat3:\n %s" % (mat1 * mat3)
    print "dot(mat1,mat3):\n %s" % numpy.dot(mat1, mat3)
    #矩阵的**就相当于矩阵与自己相乘，需要为方阵
    print "mat2 ^ 2: \n %s" % mat2 ** 2
    return

#测试array一般用途
def test_array():
    print "\tstart array test"
    arr1 = numpy.array([1, 2])
    arr2 = numpy.array([3, 4, 5])
    arr3 = numpy.array([[1, 2], [2, 2]])
    print "array1:\n %s" %arr1
    print "array2:\n %s" %arr2
    print "array3:\n %s" %arr3
    print "array3 T:\n %s" %arr3.T
    #两个array的相乘*指的是对应元素的相乘；两个array的dot表示矩阵的相乘
    print "array1 * array3:\n %s" % (arr1 * arr3)
    print "array1 dot array3:\n %s" % numpy.dot(arr1, arr3)
    #array的**相当于每个元素的平方
    print "array1 ^ 2:\n %s" % (arr1 ** 2)
    return

#测试List一般用途
def test_list():
    print "\tstart list test"
    list1 = [1, 2, 3]
    print "list1:\n %s" % list1
    return

#matrix&array
def test_matrix_array():
    print "\tstart matrix-array test"
    arr1 = numpy.array([1, 2])
    mat1 = numpy.mat('1,1;3,2')
    mat2 = numpy.mat('2, 2')
    print "array1:\n %s" % arr1
    print "mat1:\n %s" % mat1
    print "array1 * mat1:\n %s" % (arr1 * mat1)
    print "dot array1 * mat1: \n %s"% numpy.dot(arr1, mat1)
    print "array to matrix: \n%s" % numpy.mat(arr1)
    print "matrix to array: \n%s" % numpy.array(mat1)
    #下面的Matrix只能转换为array才可以运算
    print "matrix to array ^ 2:\n %s" % numpy.array(mat2) ** 2
    return

#array&list
def test_array_list():
    print "\tstart array-list test"
    arr1 = numpy.array([1, 2, 3])
    list1 = [1, 2, 3]
    print "array1, type%s value:\n %s" % (type(arr1),arr1)
    print "list1, type%s value:\n %s" % (type(list1), list1)
    print "array1 to list ret:\n%s" % arr1.tolist()
    print "list1 to array ret: \n %s"  % numpy.array(list1) 
    return

#matrix&list
def test_matrix_list():
    print"\tstart matrix&list test"
    mat1 = numpy.mat('1,2;3,4')
    list1 = [1,2]
    print "mat1:\n %s" % mat1
    print "list1:\n %s" % list1
    print "mat1 to list: \n%s" % mat1.tolist()
    print "list1 to mat:\n %s" % numpy.mat(list1)
    print "list1 * mat1: \n %s" % (list1 * mat1)
    return

if __name__ == "__main__":
    #test_matrix()
    #test_array()
    #test_list()
    test_matrix_array()
    #test_array_list()
    test_matrix_list()
