from math import log
import operator

def create_data_set():
    data_set = [[1, 1, 'yes'],
                [1, 1, 'yes'],
                [1, 0, 'no'],
                [0, 1, 'no'],
                [0, 1, 'no']]
    labels = ['no surfacing', 'flippers']
    return data_set, labels

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
        shannon_ent -= val * log(val, 2)
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

def store_tree(tree, file):
    import pickle
    fd = open(file, 'w')
    pickle.dump(tree, fd)
    fd.close()

def grab_tree(file):
    import pickle
    fd = open(file, 'r')
    return pickle.load(fd)

if __name__ == 'trees':
    print 'data_set, labels = create_data_set()'
    print 'shannon = calc_shannon_ent(data_set)'
    print 'result_set = split_data_set(data_set, axis, value)'
    print 'best_feature_pos = choose_best_feature_to_split(data_set)'
    print 'sorted_class_count = majority_cnt(class_list)'
