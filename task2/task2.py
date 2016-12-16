import sys
import numpy as np

from PIL import Image
from sklearn.cluster import KMeans
from task1 import Kmeans
import matplotlib.pyplot as plt


class InternalCriterion:
    def  __init__(self, X, clusters, centroids):
        self.X = X
        self.clusters = clusters
        self.centroids = centroids

        self.elements = np.zeros(self.centroids.shape[0])
        for i in xrange(self.X.shape[0]):
            self.elements[clusters[i]] += 1

    def davies_bouldin(self):
        msqr = np.zeros(self.centroids.shape[0])
        for i in xrange(self.X.shape[0]):
            cluster_index = self.clusters[i]
            msqr[cluster_index] += np.sum((self.centroids[cluster_index] - self.X[i]) ** 2)
            
        for i in xrange(msqr.shape[0]):
            msqr[i] /= self.elements[i]
            msqr[i] = np.sqrt(msqr[i])

        db_index = 0
        for i in xrange(self.centroids.shape[0]):
            db_index += np.max([(msqr[i] + msqr[j]) / np.linalg.norm(self.centroids[i] - self.centroids[j]) 
                          for j in xrange(self.centroids.shape[0]) if not j == i ])

        db_index /= self.centroids.shape[0]
        return db_index

    def calinski_harabasz(self):
        mean = np.mean(self.X)
        Sb = 0
        for i in xrange(self.centroids.shape[0]):
            Sb += self.elements[i] * np.sum((self.centroids[self.clusters[i]] - mean) ** 2)

        Sw = 0
        for i in xrange(self.X.shape[0]):
            Sw += np.sum((self.centroids[self.clusters[i]] - self.X[i]) ** 2)

        alpha = (self.X.shape[0] - self.centroids.shape[0]) / (self.centroids.shape[0] - 1) 
        return alpha * (Sb / Sw) 


class PairwiseCriterion:
    def  __init__(self, X, clusters_1, centroids_1, clusters_2, centroids_2):
        self.M = np.zeros((centroids_1.shape[0], centroids_2.shape[0]))
        self.elements_1 = np.zeros(centroids_1.shape[0])
        self.elements_2 = np.zeros(centroids_2.shape[0])

        self.N = X.shape[0]
        for i in xrange(self.N):
            self.M[clusters_1[i]][clusters_2[i]] += 1
            self.elements_1[clusters_1[i]] += 1
            self.elements_2[clusters_2[i]] += 1

        conv = 0
        for i in xrange(self.M.shape[0]):
            for j in xrange(self.M.shape[1]):
                conv += self.M[i][j] ** 2

        self.TP = (conv - self.N)/2
        self.FN = (np.sum(self.elements_1 ** 2) - conv) / 2
        self.FP = (np.sum(self.elements_2 ** 2) - conv) / 2
        self.TN = self.N - (self.FN + self.FP + self.TP)

    def rand(self):
        return (self.TP + self.TN) / self.N

    def fowlkes_mallows(self):
        return self.TP / np.sqrt((self.TP + self.FN) * (self.TP + self.FP))

def find_delta(chs):
    deltas = [((chs[i + 1] - chs[i]) - (chs[i] - chs[i - 1])) for i in xrange(1, len(chs) - 1)]
    return np.argmin(deltas) + 3


def image_criterions(filename, cluster_limit=10):
    print('Compressing: ' + filename)
    input_image = np.asarray(Image.open(filename))

    dbs = []
    chs = []
    nums = range(2, cluster_limit)
    for k in nums:
        print('Cluster number: ' + str(k))
        kms = Kmeans(k, 10, 1)
        img_matrix = input_image.reshape((input_image.shape[0] *  input_image.shape[1]), input_image.shape[2])
        clusters, centroids, functional = kms.fit(img_matrix)
        
        cr = InternalCriterion(img_matrix, clusters, centroids)
        dbs.append(cr.davies_bouldin())
        chs.append(cr.calinski_harabasz())

    plt.plot(nums, chs)
    plt.xlabel('Cluster Number')
    plt.ylabel('Calinski Harabasz Criterion')
    plt.grid(True)
    plt.savefig("ch_criterion.png")

    plt.clf()

    plt.plot(nums, dbs)
    plt.xlabel('Cluster Number')
    plt.ylabel('Davies Bouldin Criterion')
    plt.grid(True)
    plt.savefig("db_criterion.png")

def geom_criterions(filename, cluster_limit=10):
    file = open(filename, 'r')

    gt_clusters = []
    points = []
    for line in file.readlines():
        strs = line.strip('\n').split(' ')

        gt_clusters.append(int(strs[0]) - 1)
        points.append(np.array([float(strs[1]), float(strs[2])]))

    gt_clusters = np.array(gt_clusters)
    points = np.array(points)

    cluster_num = np.max(gt_clusters) + 1

    gt_centroids = np.zeros((cluster_num, points.shape[1]))
    gt_elements = np.zeros(cluster_num)
    for i in xrange(points.shape[0]):
        gt_centroids[gt_clusters[i]] += points[i]
        gt_elements[gt_clusters[i]] += 1

    for i in xrange(cluster_num):
        gt_centroids[i] /= gt_elements[i]

    rands = []
    fms = []
    nums = range(2, cluster_limit)
    for k in nums:
        print('Cluster number: ' + str(k))
        kms = Kmeans(k, 10, 1)
        clusters, centroids, functional = kms.fit(points)
        
        cr = PairwiseCriterion(points, gt_clusters, gt_centroids, clusters, centroids)
        rands.append(cr.rand())
        fms.append(cr.fowlkes_mallows())

    plt.plot(nums, fms)
    plt.xlabel('Cluster Number')
    plt.ylabel('Fowlkes Mallows Criterion')
    plt.grid(True)
    plt.savefig("fms_criterion.png")

    plt.clf()

    plt.plot(nums, rands)
    plt.xlabel('Cluster Number')
    plt.ylabel('Rand Criterion')
    plt.grid(True)
    plt.savefig("rand_criterion.png")

if __name__ == '__main__':
    if len(sys.argv) < 3:
        print('Error: Please, type filename and limit of clusters as command line arguments.')
        sys.exit(0)

    print(image_criterions(sys.argv[1], int(sys.argv[2])))

    
