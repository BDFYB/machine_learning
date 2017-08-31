from numpy import *
import operator

def create_data_set():
    group = array([[1.0, 1.1], [1.0, 1.0], [0, 0], [0, 0.1]])
    labels = ['A', 'A', 'B', 'B']
    return group, labels

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
    
def get_meeting_data_from_file(filename):
    try:
        fd = open(filename)
    except Exception,e:
        print "file open error!",e
        return
    lines = fd.readlines()
    line_num = len(lines)
    data = zeros((line_num, 3))
    label = []
    index = 0
    for line in lines:
        line = line.strip()
        lintrim = line.split('\t')
        data[index,:] = lintrim[0:3]
        label.append(int(lintrim[-1]))
        index += 1
    return data, label

def norm(data):
    data_min = data.min(0)
    data_max = data.max(0)
    ranges = data_max - data_min
    norm_result = zeros(shape(data))
    m = data.shape[0]
    norm_result = data - tile(data_min, (m, 1))
    norm_result = norm_result/tile(ranges, (m, 1))
    return norm_result, ranges, data_min

def dating_class_test():
    test_percentage = 0.40
    datas, labels = get_meeting_data_from_file('datingTestSet2.txt')
    data_normed, ranges, min_val = norm(datas)
    test_num = int(data_normed.shape[0] * test_percentage)
    failed_count = 0
    for i in range(test_num):
        result = knn_get_classify(data_normed[i,:], data_normed, labels, 3)
        print "test result: %d and true: %d" % (result, labels[i])
        if result != labels[i]:
            failed_count += 1
    failed_percentage = failed_count/float(test_num)
    print failed_percentage
            

if __name__ == '__main__':
    dating_class_test()

if __name__ == 'kNN':
    print 'knnGroup & knnLabel made'
    print 'method list:'
    print 'group, labels = create_data_set()'
    print 'label = knn_get_classify(point, group, labels, k)'
    print 'data, label = get_meeting_data_from_file(filename)'
    print 'norm_result, ranges, data_min = norm(data)'
    knnGroup, knnLabels = create_data_set()
