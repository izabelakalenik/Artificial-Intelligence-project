__authors__ = ['1716921', '1718541', '1720318']
__group__ = '213'

import numpy as np
import math
import operator
from scipy.spatial.distance import cdist
from collections import Counter


class KNN:
    def __init__(self, train_data, labels):
        self._init_train(train_data)
        self.labels = np.array(labels)
        #############################################################
        ##  THIS FUNCTION CAN BE MODIFIED FROM THIS POINT, if needed
        #############################################################
        self.neighbors = None

    def _init_train(self, train_data):
        """
        initializes the train data
        :param train_data: PxMxNx3 matrix corresponding to P color images
        :return: assigns the train set to the matrix self.train_data shaped as PxD (P points in a D dimensional space)
        """

        # Ensure train_data is in float format
        train_data = np.array(train_data, dtype=float)

        # Flatten each grayscale image into a 1D array
        flattened_images = train_data.reshape(train_data.shape[0], -1)

        # Assign the train set to self.train_data
        self.train_data = flattened_images

    def get_k_neighbours(self, test_data, k):
        """
        given a test_data matrix calculates de k nearest neighbours at each point (row) of test_data on self.neighbors
        :param test_data: array that has to be shaped to a NxD matrix (N points in a D dimensional space)
        :param k: the number of neighbors to look at
        :return: the matrix self.neighbors is created (NxK)
                 the ij-th entry is the j-th nearest train point to the i-th test point
        """
        # Resize test_data to match the dimensions of train_data
        resized_test_data = np.resize(test_data, self.train_data.shape)

        # Compute the distance between test_data and train_data
        distances = cdist(resized_test_data, self.train_data)

        # Get the indices of the k nearest neighbors for each sample in the test
        indices = np.argsort(distances, axis=1)[:, :k]

        # Store the labels of the k nearest neighbors
        self.neighbors = self.labels[indices]

    def get_class(self):
        """
        Get the class by maximum voting
        :return: 1 array of Nx1 elements. For each of the rows in self.neighbors gets the most voted value
                (i.e. the class at which that row belongs)
        """
        #######################################################
        ##  YOU MUST REMOVE THE REST OF THE CODE OF THIS FUNCTION
        ##  AND CHANGE FOR YOUR OWN CODE
        #######################################################
        # Initialize an empty array to store the predicted classes
        predicted_classes = np.empty(len(self.neighbors), dtype=self.neighbors.dtype)

        # Iterate over each row of neighbors
        for i, row in enumerate(self.neighbors):
            # Count the occurrences of each label while maintaining order
            label_counter = Counter(row)
            # Find the label with the maximum count
            predicted_class = max(label_counter, key=label_counter.get)
            # Store the predicted class for this row
            predicted_classes[i] = predicted_class

        return predicted_classes

    def predict(self, test_data, k):
        """
        predicts the class at which each element in test_data belongs to
        :param test_data: array that has to be shaped to a NxD matrix (N points in a D dimensional space)
        :param k: the number of neighbors to look at
        :return: the output form get_class a Nx1 vector with the predicted shape for each test image
        """

        self.get_k_neighbours(test_data, k)
        return self.get_class()
