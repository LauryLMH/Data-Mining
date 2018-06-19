""" COMP 527 CA1 Re-sit Assignment."""

from scipy.spatial.distance import euclidean
from scipy.spatial.distance import cityblock
from scipy.spatial.distance import cosine
import os.path as op

FILE_NAMES = ['test.positive', 'test.negative', 'train.negative', 'train.positive']

def ConstructFeatureSet():
    full_feature = set()
    num_words = 0
    for file in FILE_NAMES:
        with open(file) as f:
            lines = f.readlines()
            for line in lines:
                words = line.split()
                num_words += len(words)
                for w in words:
                    full_feature.add(w)
    print(len(full_feature))
    return full_feature


def ConstructBinaryVector(data, feature_set):
    bvec = []
    for f in feature_set:
        if f in data:
            bvec.append(1)
        else:
            bvec.append(0)
    return bvec


def ConstructAllDataVector():
    feature_set = ConstructFeatureSet()
    for i in range(4):
        with open(FILE_NAMES[i]) as f, open(FILE_NAMES[i]+'.out', 'w') as of:
            lines = f.readlines()
            for line in lines:
                data_list = line.split()
                of.write(str(ConstructBinaryVector(data_list, feature_set)))
                of.write('\n')
        print(FILE_NAMES[i], ' finished')


def ReadListFromFile(line):
    my_line = line[1:-2]
    ret_line = my_line.split(', ')
    return [float(i) for i in ret_line]


def ReadDataFromFile():
    train_set = []
    negative_offsite = 0
    with open('train.negative.out') as f1, open('train.positive.out') as f2:
        lines = f1.readlines()
        negative_offsite = len(lines)
        lines.extend(f2.readlines())
        for l in lines:
            train_set.append(ReadListFromFile(l))
    return negative_offsite, train_set


def TestPositive(train_set, neg_off):
    total = 0
    correct = 0
    with open('test.positive.out') as f1:
        lines = f1.readlines()
        total = len(lines)
        for j in range(total):
            sorted_tuple_list = []
            test = ReadListFromFile(lines[j])
            for i in range(len(train_set)):
                sorted_tuple_list.append((euclidean(train_set[i], test), i))
                #sorted_tuple_list.append((cityblock(train_set[i], test), i))
                #sorted_tuple_list.append((1-cosine(train_set[i], test), i))
            sorted_tuple_list.sort(key=lambda tup: tup[0])
            used_for_classification = sorted_tuple_list[:K]
            score = 0
            for w, idx in used_for_classification:
                if idx < neg_off:
                    score = score + 1
            #print(j, score)
            if score > K / 2:
                correct = correct
                #print('negative!')
            else:
                #print('positive!')
                correct = correct + 1
    return correct, total


def TestNegative(train_set, neg_off):
    total = 0
    correct = 0
    with open('test.negative.out') as f1:
        lines = f1.readlines()
        total = len(lines)
        for j in range(total):
            sorted_tuple_list = []
            test = ReadListFromFile(lines[j])
            for i in range(len(train_set)):
                sorted_tuple_list.append((euclidean(train_set[i], test), i))
                #sorted_tuple_list.append((cityblock(train_set[i], test), i))
                #sorted_tuple_list.append((1-cosine(train_set[i], test), i))
            sorted_tuple_list.sort(key=lambda tup: tup[0])
            used_for_classification = sorted_tuple_list[:K]
            score = 0
            for w, idx in used_for_classification:
                if idx < neg_off:
                    score = score + 1
            #print(j, score)
            if score > K / 2:
                #print('negative!')
                correct = correct + 1
            else:
                #print('positive!')
                correct = correct
    return correct, total

# First checking if intermediate file have been generated. If not, generate intermediate file.
# The intermediate file will cache feature vector for each test/train instance, to accelerate training process
if op.isfile(FILE_NAMES[0] + '.out'):
    print('Intermediate file ready. Directly go to training process...')
else:
    print('Intermediate file missing. Construct feature vector first...')
    ConstructAllDataVector()

neg_off, train_set = ReadDataFromFile()
# Bring back the following lines to test unbalanced training set.
#train_set = train_set[:-100]
K_array = [3, 5, 7, 9, 11]
for K in K_array:
    c1, t1 = TestPositive(train_set, neg_off)
    c2, t2 = TestNegative(train_set, neg_off)
    print(c1, c2, t1, t2)
    print('Accuracy: ', (c1+c2)/(t1+t2), 'with K = ', K)
