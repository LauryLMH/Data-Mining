
import numpy as np

MAX_ITERATIONS = 10   # largest num of iterations for the computation before convergence
TEST_CASE = 5   # num of times the computation will be performed to get the best claustering
MAX = 100000000

# Function: Main
# -------------
# Deal with function calls and output precisions
def main():
    features = []
    occurance = []
    reviews = []
    reviewLabels = []
    dataSet = []
    labels = []     # records the clauster no. datapoints belong to
    iterations = []

    # variable memory allocation for later assignments
    for i in range(0, 5):
        iterations.append(0)
        labels.append([])

    # read Data file in, assign memory location for labels
    readData(dataSet, reviews, features, occurance, reviewLabels)
    for i in range(0, 5):
        for j in range(0, len(dataSet)):
            labels[i].append(0)
    a = np.array(dataSet, dtype=np.float32)
    print 'Data Loaded.'

    # l2 normalization
    for i in range(0,len(reviews)):
        l2Length = np.linalg.norm(a[i])
        for j in range(0,len(features)):
            a[i][j] = a[i][j] / l2Length

    # Computation
    for K in range(2, 21):
        print '*********************************************'
        print 'Case ' + str(K-1) + ': K = ' + str(K)
        print '-------------------------------------------'
        print 'Using geometric mean as centroid:'
        testNo = 0
        mark = 0
        while (testNo < TEST_CASE):
            iterations[testNo] = kmeans(a, K, labels[testNo], features, mark)
            testNo += 1
        minimum = iterations[0]
        index = 0
        for i in range(1, 5):
            if minimum > iterations[i]:
                minimum = iterations[i]
                index = i
        evaluate(reviewLabels, labels[index], K)
        print '-------------------------------------------'
        print 'Using closest instance to mean as centoid: '
        mark = 1
        testNo = 0
        while (testNo < TEST_CASE):
            iterations[testNo] = kmeans(a, K, labels[testNo], features, mark)
            testNo += 1
        minimum = iterations[0]
        index = 0
        for i in range(1, 5):
            if minimum > iterations[i]:
                minimum = iterations[i]
                index = i
        evaluate(reviewLabels, labels[index], K)

################################################################################

# Function: Load Data Instances to Memory
# -------------
# Read the reviews in without the first element because
# the first element is only for evaluation purpose.
def readData(dataSet, reviews, features, occurance, reviewLabels):
    collectFeatures(features, occurance, reviewLabels)
    loadReviews(reviews)
    for i in range(0, len(reviews)):
        setOccurance(reviews[i], occurance, features)
        dataSet.append(np.copy(occurance))

def collectFeatures(features, occurance, reviewLabels):
    f = open('CA2data.txt', 'r')
    for line in f:
        line = line.strip('\n')
        reviewList = line.split(' ')
        reviewLabels.append(reviewList[0])
        for i in range(1, len(reviewList)):
            if reviewList[i] not in features:
				features.append(reviewList[i])
				occurance.append(0)

def loadReviews(reviews):
	file = open('CA2data.txt', 'r')
	for line in file:
		line = line.strip('\n')
		reviews.append(line)

def setOccurance(singleReview, occurance, features):
	reviewWords = singleReview.split(' ')
	for dataPos in range(0, len(occurance)):
		occurance[dataPos] = 0
		if features[dataPos] in reviewWords:
			occurance[dataPos] = 1

################################################################################

# Function: K Means
# -------------
# K-Means is an algorithm that takes in a dataset and a constant
# k and returns k centroids (which define clusters of data in the
# dataset which are similar to one another).
def kmeans(dataSet, K, labels, features, mark):

    length = len(features)

    # Initialize centroids randomly
    centroids = initialize_centroids(dataSet, K)

    # Initialize vars.
    iterations = 0
    oldCentroids = []

    # Run the main k-means algorithm
    while iterations < MAX_ITERATIONS:
        # Save old centroids for convergence test. Book keeping.
        oldCentroids = np.copy(centroids)
        iterations += 1

        # Assign labels to each datapoint based on centroids
        updateLabels(dataSet, centroids, labels, K)

        # Assign centroids based on datapoint labels
        updateCentroids(dataSet, labels, K, centroids, length, mark)

        # Check if any centroids changed
        if np.array_equal(oldCentroids, centroids):
            break

    return iterations


# Function: Initialize Centroids
# -------------
# Randomly choose the instances from the data set,
# return k number of instances as initial centroids.
def initialize_centroids(dataSet, K):
    # returns k centroids from the initial points
    centroids = np.copy(dataSet)
    np.random.shuffle(centroids)
    return centroids[:K]


# Function: Compute Distance
# -------------
# Returns the Euclidean distance
def distance(a, b):
    return np.linalg.norm(a-b)


# Function: Update Labels
# -------------
# Returns a label for each piece of data in the dataset with the
# same serial number.
def updateLabels(dataSet, centroids, labels, K):
    # For each element in the dataset, chose the closest centroid.
    # Make that centroid the element's label.
    for i in range(0, len(dataSet)):
        min = MAX
        for j in range(0, K):
            d = distance(dataSet[i], centroids[j])
            if  d < min:
                min = d
                labels[i] = j


# Function: Update Centroids
# -------------
# Case 1: Each centroid is the geometric mean of the points that
# have that centroid's label.
# Case 2: Each centroid is the closest instance to mean of the points
# that have the centroid's label.
def updateCentroids(dataSet, labels, K, centroids, lengthF, mark):
    # for each cluster, form the group:
    for j in range(0, K):
        # get the datapoints in the same group
        group = []
        for i in range(0, len(labels)):
            if labels[i] == j:
                group.append(dataSet[i])
        # computing geometric mean
        total = []
        for i in range(0, lengthF):
            total.append(0)
        for m in range (0, lengthF):
            for l in range(0, len(group)):
                total[m] += group[l][m]
            centroids[j][m] = total[m] / len(group)
        # case 2:
        minimum = MAX
        if mark == 1:
            for i in range(0, len(group)):
                d = distance(group[i], centroids[j])
                if d < minimum:
                    minimum = d
                    index = i
            centroids[j] = group[i]


# Function: Get Real Centroids No.
# -------------
# Returns the real clauster no. ( centroid no.) datapoint belongs to
def realCentroid(dataSet, i, centroids):
    minimum = MAX
    for j in range(0, K):
        d = distance(dataSet[i], centroids[j])
        if d < minimum:
            minimum = d
            realLabel = j
    return realLabel


# Function: Evaluate Performance.
# -------------
# Assign each cluster the label that appears most in that cluster
# Merge clausters with the same label and evaluate precision, recall and FScores
# for each label type and finally evalute the micro precision, recall and Fscore
def evaluate(reviewLabels, labels, K):
    # Vars initialization
    holder = []
    clausterLabels = []
    clausterTotal = []
    for i in range(0, K):
        clausterLabels.append(0)
        clausterTotal.append(0)
    for i in range(0, K):
        holder.append([])
        for j in range(0, 8):
            holder[i].append(0)
    # data instances dispatch into clausters
    for i in range(0, len(labels)):
        if reviewLabels[i] == 'books-positive':
            holder[labels[i]][0] += 1
        if reviewLabels[i] == 'books-negative':
            holder[labels[i]][1] += 1
        if reviewLabels[i] == 'dvd-positive':
            holder[labels[i]][2] += 1
        if reviewLabels[i] == 'dvd-negative':
            holder[labels[i]][3] += 1
        if reviewLabels[i] == 'electronics-positive':
            holder[labels[i]][4] += 1
        if reviewLabels[i] == 'electronics-negative':
            holder[labels[i]][5] += 1
        if reviewLabels[i] == 'kitchen-positive':
            holder[labels[i]][6] += 1
        if reviewLabels[i] == 'kitchen-negative':
            holder[labels[i]][7] += 1
        clausterTotal[labels[i]] += 1
    labelNum = []
    labelSet = set([])
    # counting the largest amounts of data types in each clauster
    for i in range(0, K):
        cLabel = 0
        maximum = holder[i][0]
        labelNum.append(0)
        for j in range(1, 8):
            if maximum < holder[i][j]:
                maximum = holder[i][j]
                cLabel = j
        clausterLabels[i] = cLabel
        labelSet.add(cLabel)
        labelNum[i] = holder[i][cLabel]
    # Merge clausters with same data type
    dic = {}
    counter = 0
    for cLabel in labelSet:
        counter += 1
        dic[cLabel] = []
        for i in range(0, K):
            if clausterLabels[i] == cLabel:
                dic[cLabel].append(i)
    precision = []
    recall = []
    Fscore = []
    # compute evaluation measures
    for cLabel in labelSet:
        ln = ct = 0
        for i in range(0, len(dic[cLabel])):
            ln += labelNum[dic[cLabel][i]]
            ct += clausterTotal[dic[cLabel][i]]
        precision.append(float(ln)/ct)
        recall.append(float(ln)/(51*len(dic[cLabel])))
    totalPrecision = 0
    totalRecall = 0
    totalFscore = 0
    for i in range(0, counter):
        totalPrecision += precision[i]
        totalRecall += recall[i]
        Fscore.append(2 * precision[i] * recall[i] / (precision[i] + recall[i]))
        totalFscore += Fscore[i]
    microPrecision = totalPrecision/counter
    microRecall = totalRecall/counter
    microFscore = totalFscore/counter
    # print final results 
    print 'Micro Averaged Precision:'
    print microPrecision
    print 'Micro Averaged Recall'
    print microRecall
    print 'Micro Averaged Fscore'
    print microFscore

if __name__ == "__main__":
    main()
