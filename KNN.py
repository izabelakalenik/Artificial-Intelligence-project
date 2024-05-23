__authors__ = ['1716921', '1718541', '1720318']
__group__ = '213'

import numpy as np
import math
import operator
from scipy.spatial.distance import cdist
from collections import Counter
from skimage.transform import resize
from skimage.color import rgb2hsv


class KNN:
    def __init__(self, train_data, labels, feature_method=2,downsample=False, downsample_factor=2):
        # self._init_train(train_data)
        self.feature_method = feature_method
        self.downsample = downsample
        self.downsample_factor = downsample_factor
        self._init_train_advanced(train_data)
 
        self.labels = np.array(labels)
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
        # Initial train data (P, M, N, 3) , after reshape (P, M * N * 3)
        flattened_images = train_data.reshape(train_data.shape[0], -1)

        # Assign the train set to self.train_data
        self.train_data = flattened_images

    def _init_train_advanced(self, train_data):
        """
        initializes the train data
        :param train_data: PxMxNx3 matrix corresponding to P color images
        :return: assigns the train set to the matrix self.train_data shaped as PxD (P points in a D dimensional space)
        """
        train_data = np.array(train_data, dtype=float)
        processed_data = [self._extract_features(img) for img in train_data]
        self.train_data = np.array(processed_data)
        print(f"Training data shape after feature extraction: {self.train_data.shape}")

    def _extract_features(self, img):
        if self.feature_method == 1:
            # Flatten the image with optional resizing
            if self.downsample:
                img = img[::self.downsample_factor, ::self.downsample_factor]
            return img.reshape(-1)
        elif self.feature_method == 2:
            # Extract custom features: mean, variance, max, min pixel values
            # mean_val = np.mean(img)
            # var_val = np.var(img)
            # upper_val = np.max(img)
            # lower_val = np.min(img)
            # return [mean_val, var_val, upper_val, lower_val]
            hsv_img = rgb2hsv(img)
            hue_mean = np.mean(hsv_img[:, :, 0])
            saturation_mean = np.mean(hsv_img[:, :, 1])
            brightness_mean = np.mean(hsv_img[:, :, 2])
            return [hue_mean, saturation_mean, brightness_mean]
        else:
            raise ValueError("Invalid feature method specified")


    def get_k_neighbours(self, test_data, k):
        """
        given a test_data matrix calculates de k nearest neighbours at each point (row) of test_data on self.neighbors
        :param test_data: array that has to be shaped to a NxD matrix (N points in a D dimensional space)
        :param k: the number of neighbors to look at
        :return: the matrix self.neighbors is created (NxK)
                 the ij-th entry is the j-th nearest train point to the i-th test point
        """
        # Resize test_data to match the dimensions of train_data
        flattened_testdata = test_data.reshape(test_data.shape[0], -1)
        #resized_test_data = np.resize(test_data, self.train_data.shape)
        # Compute the distance between test_data and train_data
        distances = cdist(flattened_testdata, self.train_data)

        # Get the indices of the k nearest neighbors for each sample in the test
        indices = np.argsort(distances, axis=1)[:, :k]

        # Store the labels of the k nearest neighbors
        self.neighbors = self.labels[indices]


    def get_k_neighbours_advanced(self, test_data, k):
        # Extract features for each image in the test set
        processed_test_data = np.array([self._extract_features(img) for img in test_data])
        print(f"Test data shape after feature extraction: {processed_test_data.shape}")

        distances = cdist(processed_test_data, self.train_data)
        indices = np.argsort(distances, axis=1)[:, :k]
        self.neighbors = self.labels[indices]    

    def get_class(self):
        """
        Get the class by maximum voting
        :return: 1 array of Nx1 elements. For each of the rows in self.neighbors gets the most voted value
                (i.e. the class at which that row belongs)
        """
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

        self.get_k_neighbours_advanced(test_data, k)
        return self.get_class()
