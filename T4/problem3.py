# CS 181, Spring 2017
# Homework 4: Clustering
# Name:
# Email:

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

class KMeans(object):
    # K is the K in KMeans
    def __init__(self, K):
        self.K = K

    # X is a (N x 28 x 28) array where 28x28 is the dimensions of each of the N
    # images.
    def fit(self, X):
        self.N = X.shape[0]
        self.assignments = [None for i in range(self.N)]
        self.clusters = [set() for i in range(self.K)]
        self.means = X[np.random.choice(pics.shape[0], size=self.K, replace=False),:,:]
        converged = False
        steps = 0
        self.losses = []
        counter = 0
        while not converged:
            print(counter)
            counter += 1
            converged = True
            loss = 0
            for i in range(self.N):
                dists = [np.linalg.norm(X[i,:,:] - mean) for mean in self.means]
                cluster = np.argmin(dists)
                loss += min(dists)
                if self.assignments[i] != cluster:
                    self.clusters[cluster].add(i)
                    if self.assignments[i] is not None:
                        self.clusters[self.assignments[i]].discard(i)
                    self.assignments[i] = cluster
                    converged = False
            self.means = [np.mean(X[list(a),:,:], axis=0) for a in self.clusters]
            self.losses.append(loss)
            steps += 1
            if converged:
                break
        print('K-means converged in %i steps' % (steps))

    # This should return the arrays for K images. Each image should represent
    # the mean of each of the fitted clusters.
    def get_mean_images(self):
        return self.means

    # This should return the arrays for D images from each cluster that are
    # representative of the clusters.
    def get_representative_images(self, D, X):
        reps = []
        for c in range(self.K):
            arr = np.argsort([np.linalg.norm(X[pt,:,:] - self.means[c]) for pt in self.clusters[c]])[:D]
            reps.append(X[arr,:,:])
        return reps

    # img_array should be a 2D (square) numpy array.  Note, you are welcome to
    # change this function (including its arguments and return values) to suit
    # your needs.  However, we do ask that any images in your writeup be
    # grayscale images, just as in this example.
    def create_image_from_array(self, img_array, prefix=''):
        for e, img in enumerate(img_array):
            title = 'figures/' + prefix + str(e) + '.png'
            plt.figure()
            plt.imshow(img, cmap='Greys_r')
            plt.savefig(title, bbox_inches='tight')
        return

    def plot_loss(self):
        plt.figure()
        plt.plot(range(len(self.losses)), self.losses)
        plt.xlabel('epoch')
        plt.ylabel('kmeans loss')
        plt.savefig('figures/loss.png', bbox_inches='tight')
        return

# This line loads the images for you. Don't change it!
pics = np.load("images.npy", allow_pickle=False)

# You are welcome to change anything below this line. This is just an example
# of how your code may look.  That being said, keep in mind that you should not
# change the constructor for the KMeans class, though you may add more public
# methods for things like the visualization if you want.  Also, you must
# cluster all of the images in the provided dataset, so your code should be
# fast enough to do that.
K = 10
KMeansClassifier = KMeans(K=K)
KMeansClassifier.fit(pics)
KMeansClassifier.create_image_from_array(KMeansClassifier.get_mean_images(), prefix='k3_centroids/')
KMeansClassifier.plot_loss()
reps = KMeansClassifier.get_representative_images(3, pics)
for e, reps_num in enumerate(reps):
    KMeansClassifier.create_image_from_array(reps_num, prefix='k3_repr/' + str(e) + '_')
# KMeansClassifier = KMeans(K=10, useKMeansPP=False)
# KMeansClassifier.fit(pics)
# KMeansClassifier.create_image_from_array(pics[1])
