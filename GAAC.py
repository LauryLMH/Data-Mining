from scipy.spatial.distance import euclidean
import os.path as op
import time

FILE_NAME = 'CA2data.txt'
INTERMEDIATE_FILE_NAME = 'distances.txt'
N = 1000


def ReadData(N):
    """Read data into data structure and get all feature set"""
    word_count = {}
    test_set = []
    with open(FILE_NAME) as f:
        lines = f.readlines()
        for l in lines:
            words = l.split()
            correct_cluster = words[0]
            features = words[1:]
            test_set.append((correct_cluster, features))
            for f in features:
                if f not in word_count:
                    word_count[f] = 0
                word_count[f] += 1
    feature_list = sorted(word_count.items(), key=lambda x: x[1], reverse=True)
    print('Trunk vocaburary set to ', N)
    feature_list = feature_list[:N]
    feature_set = set([f[0] for f in feature_list])
    print(feature_set)
    print('The size of feature set: ', len(feature_set))
    print('The size of test set: ', len(test_set))
    return feature_set, test_set


def CalculateDistances(feature_set, test_set):
    """Calculate distances between test instances. Write to file for future usage"""
    distances = []
    test_features = []
    for t in test_set:
        features = t[1]
        t_feature = []
        for f in feature_set:
            if f in features:
                t_feature.append(1)
            else:
                t_feature.append(0)
        test_features.append(t_feature)
    for i in range(len(test_features)):
        dist = []
        for j in range(len(test_features)):
            dist.append(euclidean(test_features[i], test_features[j]))
        distances.append(dist)
    # Write distances to file
    with open(INTERMEDIATE_FILE_NAME, 'w') as f:
        for dist in distances:
            for d in dist:
                f.write(str(d) + ' ')
            f.write('\n')

    return distances


def ReadDistanceFile():
    distances = []
    with open(INTERMEDIATE_FILE_NAME) as f:
        lines = f.readlines()
        for l in lines:
            d = l.split()
            distances.append([float(di) for di in d])
    return distances


def GetDistance(N):
    if not op.isfile(INTERMEDIATE_FILE_NAME):
        print('No distances file. Creating new one...')
        feature_set, test_set = ReadData(N)
        return CalculateDistances(feature_set, test_set)
    else:
        print('Already found distances file. Reading...')
        return ReadDistanceFile()


def GAACDistance(distances, l1, l2):
    ret = 0
    for l11 in l1:
        for l22 in l2:
            ret += distances[l11][l22]
    return ret/len(l1)/len(l2)


def CompleteLinkageDistance(distances, l1, l2):
    ret = 0
    for l11 in l1:
        for l22 in l2:
            if distances[l11][l22] > ret:
                ret = distances[l11][l22]
    return ret


def FindNextClusterToMerge(distances, clusters, way):
    csize = len(clusters)
    min_distance = 1000
    return_set = None
    for i in range(csize):
        j = i + 1
        while j < csize:
            if way is True:
                d = GAACDistance(distances, clusters[i], clusters[j])
            else:
                d = CompleteLinkageDistance(distances, clusters[i], clusters[j])
            # print('distance between ', clusters[i], ' and ', clusters[j], ' is: ', d)
            if d < min_distance:
                min_distance = d
                return_set = (i, j)
            j += 1
    #print(min_distance)
    return return_set


def MergeCluster(clusters, i, j):
    clusters[i].extend(clusters[j])
    del clusters[j]
    return clusters


def Cluster(distances, ifGAAC):
    num_test = len(distances)
    # Construct initial clusters
    clusters = []
    for i in range(num_test):
        clusters.append([i])
    while len(clusters) != 8:
        i, j = FindNextClusterToMerge(distances, clusters, ifGAAC)
        #print('merge ', clusters[i], ' ', clusters[j])
        clusters = MergeCluster(clusters, i, j)
        #print(len(clusters))
    return clusters


distances = GetDistance(N)
start = time.time()
# True for GAACDistance, False for CompleteLinkageDistance
clusters = Cluster(distances, True)
end = time.time()
print('Elapsed time: ', end-start)
