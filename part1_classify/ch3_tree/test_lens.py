import trees
import pickle

if __name__ == "__main__":
    fd = open('lenses.txt')
    lenses = [inst.strip().split('\t') for inst in fd.readlines()]
    #print lenses
    lenses_feature = ['age', 'prescript', 'astigmatic', 'tear_rate']
    len_tree = trees.create_tree(lenses, lenses_feature)
    print len_tree
    result = 'lens_tree'
    fdl = open(result, 'w')
    pickle.dump(len_tree, fdl)
    fdl.close()
