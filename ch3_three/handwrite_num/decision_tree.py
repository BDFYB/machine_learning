
from math import log
from numpy import *
import os
import operator
import pickle

def calc_shannon_ent(data_set):
    num_entries = len(data_set)
    label_counts = {}
    for fet_vec in data_set:
        current_label = fet_vec[-1]
        if current_label not in label_counts.keys():
            label_counts[current_label] = 0
        label_counts[current_label] += 1
    shannon_ent = 0.0
    for key in label_counts:
        val = float(label_counts[key]) / num_entries
        #numpy has log method too, if use log(val, 2) directly will be wrong
        shannon_ent -= val * math.log(val, 2)
    return shannon_ent

def split_data_set(data_set, axis, value):
    ret_data_set = []
    for feat_vec in data_set:
        if feat_vec[axis] == value:
            reduced_feat_vec = feat_vec[:axis]
            reduced_feat_vec.extend(feat_vec[axis + 1 :])
            ret_data_set.append(reduced_feat_vec)
    return ret_data_set

def choose_best_feature_to_split(data_set):
    num_features = len(data_set[0]) -1
    base_entropy = calc_shannon_ent(data_set)
    best_info_gain = 0.0
    best_feature = -1
    for i in range(num_features):
        feat_list = [example[i] for example in data_set]
        unique_vals = set(feat_list)
        new_entropy = 0.0
        for value in unique_vals:
            sub_data_set = split_data_set(data_set, i, value)
            prob = len(sub_data_set) / float(len(data_set))
            new_entropy += prob * calc_shannon_ent(sub_data_set)
        info_gain = base_entropy - new_entropy
        if info_gain > best_info_gain:
            best_info_gain = info_gain
            best_feature = i
    return best_feature

def majority_cnt(class_list):
    class_count = {}
    for vote in class_list:
        if vote not in class_count.keys():
            class_count[vote] = 0
        class_count[vote] += 1
    sorted_class_count = sorted(class_count.iteritems(), key = operator.itemgetter(1), reverse = true)
    return sorted_class_count

def create_tree(data_set, labels):
    tmp_labels = labels[:]
    class_list = [example[-1] for example in data_set]
    if class_list.count(class_list[0]) == len(class_list):
        return class_list[0]
    if len(data_set[0]) == 1:
        return majority_cnt(class_list)
    best_feat = choose_best_feature_to_split(data_set)
    best_feat_label = tmp_labels[best_feat]
    my_tree = {best_feat_label:{}}
    del(tmp_labels[best_feat])
    feat_values = [example[best_feat] for example in data_set]
    unique_vals = set(feat_values)
    for value in unique_vals:
        sub_labels = tmp_labels[:]
        my_tree[best_feat_label][value] = create_tree(split_data_set(data_set, best_feat, value), sub_labels)
    return my_tree   

def classify(input_tree, feat_labels, test_vec):
    first_node = input_tree.keys()[0]
    second_dict = input_tree[first_node]
    feat_index = feat_labels.index(first_node)
    for key in second_dict:
        if test_vec[feat_index] == key:
            if type(second_dict[key]).__name__ == 'dict':
                class_label = classify(second_dict[key], feat_labels, test_vec)
            else:
                class_label = second_dict[key]
    return class_label

def get_num_data():
    dir_name = 'trainingDigits'
    file_list = os.listdir('trainingDigits')
    file_num = len(file_list)
    data = []

    for i in range(file_num):
        #label = int(file_list[i].split('_')[0])
        label = file_list[i].split('_')[0]
        single_data = []
        fd = open('trainingDigits/%s' % file_list[i])
        for x in range(32):
            line = fd.readline()
            for y in range(32):
                #single_data.append(int(line[y]))  
                single_data.append(line[y])
        single_data.append(label)
        data.append(single_data) 
    return data

def get_labels():
    labels = []
    for j in range(1024):
        labels.append('p%s' % j)
    return labels

def test_pic_to_num(file_name):
    fd = open("testDigits/%s" % file_name)
    single_data = []
    for x in range(32):
        line = fd.readline()
        for y in range(32):
            #single_data.append(int(line[y]))    
            single_data.append(line[y])
    single_data.append(file_name.split('_')[0])
    return single_data

def serialize_tree(data_tree):
    fd = open("trained_tree", 'w')
    pickle.dump(data_tree, fd)
    fd.close()    

def load_tree():
    fd = open("trained_tree", 'r')
    data = pickle.load(fd)
    fd.close() 
    return data

if __name__ == '__main__':
    #data_set = get_num_data()
    labels = get_labels()
    #data_tree = create_tree(data_set, labels)
    #serialize_tree(data_tree)
    data_tree = load_tree()

    file_list = os.listdir("testDigits")
    file_num = len(file_list)
    wrong_count = 0
    for i in range(file_num):
        real_num = file_list[i].split('_')[0]
        target_vec = test_pic_to_num(file_list[i])
        recognize_as = classify(data_tree, labels, target_vec)
        if recognize_as != real_num:
            wrong_count += 1
            print "%s recognized as: %s, file: testDigits/%s" % (real_num, recognize_as, file_list[i])
    failed_percentage = float(wrong_count) / file_num
    print failed_percentage
    #target_vec = test_pic_to_num("7_0.txt")
    #recognize_as = classify(data_tree, labels, target_vec)


