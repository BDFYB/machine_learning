from numpy import *
import operator
import os

def knn_get_classify(point, group, labels, k):
    size = group.shape[0]
    calculate = tile(point, (size, 1)) - group
    dist = calculate ** 2
    distance = dist.sum(axis = 1)
    distance = distance ** 0.5
    indexes = distance.argsort()
    counts = {}
    for i in range(k):
        ilabel = labels[indexes[i]]
        counts[ilabel] = counts.get(ilabel, 0) + 1
    results = sorted(counts.iteritems(), key = operator.itemgetter(1), reverse = True)
    return results[0][0]
    
def pic_to_num(file_name):
    matrix = zeros((1, 1024))
    fd = open(file_name)
    for i in range(32):
        line = fd.readline()
        for j in range(32):
            matrix[0, 32 * i + j] = int(line[j])
    return matrix

def get_trainning_data_from_dir():
    trainning_file_list = os.listdir('trainingDigits')
    file_num = len(trainning_file_list)
    data = zeros((file_num, 1024))
    labels = []
    for i in range(file_num):
        label = int(trainning_file_list[i].split('_')[0])
        labels.append(label)
        data[i, :] = pic_to_num("trainingDigits/%s" % trainning_file_list[i])
    return data, labels

def handwrite_num_test():
    tranning_data, tranning_labels = get_trainning_data_from_dir()
    test_file_list = os.listdir('testDigits')
    file_num = len(test_file_list)
    fail_count = 0
    for i in range(file_num):
        actual = int(test_file_list[i].split('_')[0])
        pic = pic_to_num("testDigits/%s" % test_file_list[i])
        pridict = knn_get_classify(pic, tranning_data, tranning_labels, 3)
        if pridict != actual:
            fail_count += 1
            print "real num %d, recognize as %d, file: testDigits/%s" % (actual, pridict, test_file_list[i])
        #else:
            #print "file: %s, recognize: %d" % (test_file_list[i], pridict)
    fail_percentage = fail_count / float(file_num)
    print "wrong percentage: %f" % fail_percentage


            

if __name__ == '__main__':
    handwrite_num_test()

if __name__ == 'kNN':
    print 'knnGroup & knnLabel made'
    print 'method list:'
    print 'group, labels = create_data_set()'
    print 'label = knn_get_classify(point, group, labels, k)'
    print 'data, label = get_meeting_data_from_file(filename)'
    print 'norm_result, ranges, data_min = norm(data)'
