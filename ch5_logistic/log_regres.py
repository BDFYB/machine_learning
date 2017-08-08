

def load_dataset():
    data_mat = []
    label_mat = []
    fd = open('test_set.txt')
    for line in fd.readlines():
        line_sp = line.strip().split()
        data_mat.append([1.0, float(line_sp[0]), float(line_sp[1])])
        label_mat.append(line_sp[2])
    return data_mat, label_mat


if __name__ == "__main__":
    data_mat, label_mat = load_dataset()
    print data_mat
    print label_mat

