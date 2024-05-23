__authors__ = ['1716921', '1718541', '1720318']
__group__ = 'noneyet'

import numpy as np
import utils
import copy
from math import floor

class KMeans:

    def __init__(self, X, K=1, options=None):
        """
         Constructor of KMeans class
             Args:
                 K (int): Number of cluster
                 options (dict): dictionary with options
            """
        self.num_iter = 0
        self.K = K
        self._init_X(X)
        self._init_options(options)  # DICT options

        #############################################################
        ##  THIS FUNCTION CAN BE MODIFIED FROM THIS POINT, if needed
        #############################################################
        self.labels = None
        self.centroids = None
        self.old_centroids = None

    def _init_X(self, X):
        """Initialization of all pixels, sets X as an array of data in vector form (PxD)
            Args:
                X (list or np.array): list(matrix) of all pixel values
                    if matrix has more than 2 dimensions, the dimensionality of the sample space is the length of
                    the last dimension
        """

        Y = X.astype(float)
        num_rows, num_col, dimension = Y.shape
        N = num_rows * num_col
        Y = np.reshape(Y, (N, 3))
        self.X = Y

    def _init_options(self, options=None):
        """
        Initialization of options in case some fields are left undefined
        Args:
            options (dict): dictionary with options
        """
        if options is None:
            options = {}
        if 'km_init' not in options:
            options['km_init'] = 'first'
        if 'verbose' not in options:
            options['verbose'] = False
        if 'tolerance' not in options:
            options['tolerance'] = 0
        if 'max_iter' not in options:
            options['max_iter'] = np.inf
        if 'fitting' not in options:
            options['fitting'] = 'WCD'  # within class distance.
        if 'threshold' not in options:
            options['threshold'] = 20

            # If your methods need any other parameter you can add it to the options dictionary
        self.options = options

        #############################################################
        ##  THIS FUNCTION CAN BE MODIFIED FROM THIS POINT, if needed
        #############################################################

    def _init_centroids(self):
        """
        Initialization of centroids
        """

        if self.options['km_init'].lower() == 'first':
            unique_elements, index = np.unique(self.X, axis=0, return_index=True)
            self.centroids = np.array([self.X[i] for i in sorted(index)[:self.K]])
        elif self.options['km_init'].lower() == 'random':
            self.centroids = np.random.rand(self.K, self.X.shape[1])
        elif self.options['km_init'].lower() == 'custom':
            diagonal_points = np.zeros((self.K, self.X.shape[1]))
            for i in range(self.K):
                diagonal_points[i] = np.full((self.X.shape[1],), i)
            self.centroids = diagonal_points
        elif self.options['km_init'].lower() == 'naive_sharding':
            data_set = copy.deepcopy(self.X)
            self.centroids = np.mat(np.zeros((self.K, data_set.shape[1])))
            composite = np.mat(np.sum(data_set, axis=1))
            data_set = np.append(composite.T, data_set, axis=1)
            data_set.sort(axis=0)

            step = floor(data_set.shape[0]/self.K)
            vfunc = np.vectorize(get_mean)

            for i in range(self.K):
                if i == self.K-1:
                    self.centroids[i:] = vfunc(np.sum(data_set[i*step:, 1:], axis=0), step)
                else:
                    self.centroids[i:] = vfunc(np.sum(data_set[i * step:(i+1)*step, 1:], axis=0), step)
                self.centroids = np.asarray(self.centroids)
        elif self.options['km_init'].lower() == 'plus_plus':
            self.centroids = np.random.rand(self.K, self.X.shape[1])
            for k in range(1, self.K):
                dist_sq = np.array([min([np.inner(c-x,c-x) for c in self.centroids]) for x in self.X])
                probs = dist_sq / dist_sq.sum()
                cumulative_probs = probs.cumsum()
                r = np.random.rand()
                for j, p in enumerate(cumulative_probs):
                    if r < p:
                        index = j
                        self.centroids[k] = self.X[index]
                        break

        self.old_centroids = copy.deepcopy(self.centroids)

    def get_labels(self):
        """
        Calculates the closest centroid of all points in X and assigns each point to the closest centroid
        """

        distances = distance(self.X, self.centroids)
        self.labels = np.argmin(distances, axis=1)

    def get_centroids(self):
        """
        Calculates coordinates of centroids based on the coordinates of all the points assigned to the centroid
        """

        self.old_centroids = np.copy(self.centroids)

        for i in range(self.K):
            self.centroids[i] = np.mean(self.X[self.labels == i], axis=0)

    def converges(self):
        """
        Checks if there is a difference between current and old centroids
        """

        if np.allclose(self.centroids, self.old_centroids, atol=self.options['tolerance']):
            return True
        return False

    def fit(self):
        """
        Runs K-Means algorithm until it converges or until the number of iterations is smaller
        than the maximum number of iterations.
        """

        self.num_iter = 0
        self._init_centroids()
        while self.num_iter < self.options['max_iter']:
            self.get_labels()
            self.get_centroids()
            self.num_iter += 1
            if self.converges():
                break
        return self.centroids

    def withinClassDistance(self):
        """
         returns the within class distance of the current clustering
         within-class distance measures the compactness of the clusters
        """

        wcd = 0
        for i in range(self.K):
            cluster_points = self.X[self.labels == i]  # Points in current cluster
            centroid = self.centroids[i]  # Centroid of current cluster

            # Sum of squared distances from each point to the centroid
            sum_squared_distances = np.sum((cluster_points - centroid) ** 2)
            wcd += sum_squared_distances

        # Divide by total nr of points in all K clusters
        wcd /= len(self.X)
        return wcd
    
    def interClassDistance(self):
        """
        returns the inter class distance of the current clustering
        inter-class distance measures how well-separated the clusters are
        """
        inter_class_dist = []
        num_pairs = 0
        for i in range(self.K):
            for j in range(i + 1, self.K):
                labels_i = self.X[self.labels == i]
                expanded = labels_i[:, np.newaxis, :]
                distances = np.linalg.norm(expanded - self.X[self.labels == j], axis=-1)
                inter_class_dist.append(np.mean(distances))
        return np.mean(inter_class_dist)

    def fisherDiscriminant(self):
        """
         returns the fisherDiscriminant of the current clustering
         gives a ratio of the compactness of the clusters to their separation
        """
        intra_class_dist = self.withinClassDistance()
        inter_class_dist = self.interClassDistance()
        return intra_class_dist / inter_class_dist

    def find_bestK(self, max_K):
        """
         sets the best k analysing the results up to 'max_K' clusters
        """

        wcd_values = []
        for k in range(2, max_K + 1):
            self.K = k
            self.fit()
            wcd = self.withinClassDistance()

            if k > 2:
                pct_decrease = 100 * (1 - (wcd / wcd_values[-1]))
                if pct_decrease < self.options['threshold']:
                    self.K = k - 1
                    return k - 1  # Return last k that was above the threshold
            wcd_values.append(wcd)

        # If decrease never fell below the threshold
        return max_K
    

def distance(X, C):
    """
    Calculates the distance between each pixel and each centroid
    Args:
        X (numpy array): PxD 1st set of data points (usually data points)
        C (numpy array): KxD 2nd set of data points (usually cluster centroids points)

    Returns:
        dist: PxK numpy array position ij is the distance between the
        i-th point of the first set and the j-th point of the second set
    """

    return np.linalg.norm(X[:, None] - C, axis=-1)


def get_colors(centroids):
    """
    for each row of the numpy matrix 'centroids' returns the color label following the 11 basic colors as a LIST
    Args:
        centroids (numpy array): KxD 1st set of data points (usually centroid points)

    Returns:
        labels: list of K labels corresponding to one of the 11 basic colors
    """

    color_probabilities = utils.get_color_prob(centroids)
    max_prob_index = np.argmax(color_probabilities, axis=1)
    color_labels = utils.colors[max_prob_index]

    return color_labels


def get_mean(sums, step):
    return sums/step
