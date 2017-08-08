import numpy

#make data
def load_data_set():
    posting_list =[['my', 'dog', 'has', 'flea', 'problems', 'help', 'please'],
                 ['maybe', 'not', 'take', 'him', 'to', 'dog', 'park', 'stupid'],
                 ['my', 'dalmation', 'is', 'so', 'cute', 'I', 'love', 'him'],
                 ['stop', 'posting', 'stupid', 'worthless', 'garbage'],
                 ['mr', 'licks', 'ate', 'my', 'steak', 'how', 'to', 'stop', 'him'],
                 ['quit', 'buying', 'worthless', 'dog', 'food', 'stupid']]
    class_vec = [0,1,0,1,0,1]    
    return posting_list, class_vec

#make word unrepeated in a text
def create_vocabl_list(data_set):
    vocab_set = set([])
    for document in data_set:
        vocab_set = vocab_set | set(document)
    return list(vocab_set)

#determine text words what in vocab_list
def set_of_words_2_vec(vocab_list, input_set):
    return_vec = [0] * len(vocab_list)
    for word in input_set:
        if word in vocab_list:
            return_vec[vocab_list.index(word)] = 1
        else:
            print "the word: %s is not in my vocabulary!" % word
    return return_vec

def train_nb0(train_matrix, train_category):
    num_train_docs = len(train_matrix)
    num_words = len(train_matrix[0])
    p_abusive = sum(train_category) / float(num_train_docs)
    p0_num = numpy.ones(num_words)
    p1_num = numpy.ones(num_words)
    p0_denom = 2.0
    p1_denom = 2.0
    for i in range(num_train_docs):
        if train_category[i] == 1:
            p1_num += train_matrix[i]
            p1_denom += sum(train_matrix[i])
        else:
            p0_num += train_matrix[i]
            p0_denom += sum(train_matrix[i])
    p1_vec = numpy.log(p1_num / p1_denom)
    p0_vec = numpy.log(p0_num / p0_denom)
    return p0_vec, p1_vec, p_abusive

def classify_using_nb(target_vec, p0_vec, p1_vec, p1_probility):
    p1 = sum(target_vec * p0_vec) + numpy.log(p1_probility)
    p0 = sum(target_vec * p1_vec) + numpy.log(1 - p1_probility)
    if p1 > p0:
        return 1
    else:
        return 0

    

if __name__ == "__main__":
    posting_list, class_vec = load_data_set()
    vocab_list = create_vocabl_list(posting_list)
    train_matrix = []
    for current in posting_list:
        train_matrix.append(set_of_words_2_vec(vocab_list, current))
    p0_v, p1_v, p_ab = train_nb0(train_matrix, class_vec)
    """
    print vocab_list
    print train_matrix
    print p0_v 
    print p1_v 
    print p_ab
    """
    test_word1 = ['love','my','dalmation']
    test_word2 = ['stupid','garbage']
    test_vec1 = set_of_words_2_vec(vocab_list, test_word1)
    test_vec2 = set_of_words_2_vec(vocab_list, test_word2)
    result1 = classify_using_nb(test_vec1, p0_v, p1_v, p_ab)
    result2 = classify_using_nb(test_vec2, p0_v, p1_v, p_ab)
    print test_word1, "classified as: %s" % result1
    print test_word2, "classified as: %s" % result2

