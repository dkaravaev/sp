import sys
import numpy as np

from PIL import Image
from sklearn.cluster import KMeans

class Kmeans:
    def __init__(self, cluster_num, iter_num=10, reuse=1):
        self.cluster_num = cluster_num
        self.iter_num = iter_num
        self.reuse = reuse

        self.values = []
        self.best_value_ind = 0

    def fit_raw(self, X):
        clusters = np.zeros(X.shape[0], dtype=int)
        centroids = np.array([X[i] for i in np.random.randint(X.shape[0], size=self.cluster_num)])

        functional = 0
        for i in xrange(self.iter_num):
            newcentroids = np.zeros((self.cluster_num, X.shape[1]))
            elements = np.zeros(self.cluster_num)

            for j in range(X.shape[0]):
                norms = np.array([np.linalg.norm(centroid - X[j]) for centroid in centroids])
                clusters[j] = np.argmin(norms)

                newcentroids[clusters[j]] += X[j]
                elements[clusters[j]] += 1

                functional += norms[clusters[j]] ** 2 

            for j in xrange(self.cluster_num):
                if not elements[j] == 0:
                    newcentroids[j] /= elements[j]

            if np.equal(centroids, newcentroids).all():
                return clusters, centroids, functional

            centroids = newcentroids

        return clusters, centroids, functional

    def fit(self, X):
        self.values = [self.fit_raw(X) for i in xrange(self.reuse)]
        self.best_value_ind = np.argmin([value[2] for value in self.values])

        return self.values[self.best_value_ind] 


def compress_original(input_image, k, n, reuse):
    kms = Kmeans(k, n, reuse)

    img_matrix = input_image.reshape((input_image.shape[0] *  input_image.shape[1]), input_image.shape[2])
    clusters, centroids, functional = kms.fit(img_matrix)
    
    output_image = np.zeros(img_matrix.shape)
    for i in xrange(img_matrix.shape[0]):
        output_image[i] = centroids[clusters[i]]
        
    return output_image.reshape(input_image.shape).astype(np.uint8)

def compress_sklearn(input_image, k):
    img_matrix = input_image.reshape((input_image.shape[0] * input_image.shape[1]), input_image.shape[2])
    kmeans = KMeans(n_clusters=k, random_state=0).fit(img_matrix)

    output_image = np.zeros(img_matrix.shape)
    for i in xrange(img_matrix.shape[0]):
        output_image[i] = kmeans.cluster_centers_[kmeans.labels_[i]]

    return output_image.reshape(input_image.shape).astype(np.uint8)

if __name__ == '__main__':
    image_name = 'policemen.jpg'

    print('Compressing: ' + image_name)
    input_image = np.asarray(Image.open(image_name))

    comp_db = compress_original(input_image, 7, 10, 1)
    comp_ch = compress_original(input_image, 8, 10, 1)

    Image.fromarray(comp_db).save('db_compressed_' + image_name)
    Image.fromarray(comp_ch).save('ch_compressed_' + image_name)
