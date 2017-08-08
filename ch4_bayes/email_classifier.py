import numpy
import re
import os
import random

def text_parser(big_string):
    list_of_tokens = re.split(r'\W*', big_string)
    return [tok.lower() for tok in list_of_tokens if len(tok) > 2]

def create_vocabl_list(data_set):
    vocab_set = set([])
    for document in data_set:
        vocab_set = vocab_set | set(document)
    return list(vocab_set)

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
    p1 = sum(target_vec * p1_vec) + numpy.log(p1_probility)
    p0 = sum(target_vec * p0_vec) + numpy.log(1 - p1_probility)
    if p1 > p0:
        return 1
    else:
        return 0

def bagOfWords2VecMN(vocabList, inputSet):
    returnVec = [0]*len(vocabList)
    for word in inputSet:
        if word in vocabList:
            returnVec[vocabList.index(word)] += 1
    return returnVec

if __name__ == "__main__":
	doc_list = []
	class_list = []
	full_text = []
	spam_files = os.listdir("./email/spam")
	for one_file in spam_files:
		word_list = text_parser(open("./email/spam/%s"% one_file).read())
		doc_list.append(word_list)
		full_text.extend(word_list)
		class_list.append(1)
	ham_files = os.listdir("./email/ham")
	for one_file in ham_files:
		word_list = text_parser(open("./email/ham/%s" % one_file).read())
		doc_list.append(word_list)
		full_text.extend(word_list)	
		class_list.append(0)

	vocab_list = create_vocabl_list(doc_list)

	trainning_set = range(len(doc_list))
	test_set = []
	for i in range(10):
		random_index = int(random.uniform(0,len(trainning_set)))
		test_set.append(trainning_set[random_index])
		del(trainning_set[random_index])

	train_matrix = []
	train_class = []
	for index in trainning_set:
		train_matrix.append(set_of_words_2_vec(vocab_list, doc_list[index]))
		train_class.append(class_list[index])
	p0_v, p1_v, p_spam = train_nb0(train_matrix, train_class)

	error_count = 0
	for index in test_set:
		test_vec = set_of_words_2_vec(vocab_list, doc_list[index])
		result = classify_using_nb(test_vec, p0_v, p1_v, p_spam)
		if result != class_list[index]:
			error_count += 1

	print "error num: ", error_count
	print 'the error rate is: ',float(error_count)/len(test_set)















