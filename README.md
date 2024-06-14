# Artificial-Intelligence-project

Project for the Artificial Intelligence course - 6th semester, completed in a group of 3 people.
 <br />
 <br />

## Automated Image Labeling for Smart Searches on an Online Shop
This project aims to build an intelligent agent capable of automatically labeling images of clothing items to enable smart searches using natural language on an online shop. 
The system assigns two types of labels to new products: Color and Shape. This labeling task is achieved through the implementation of two distinct algorithms: K-means and K-Nearest Neighbors (KNN).
Users should be able to search for: “Red Shirt” or “Black Sandals”.

The primary goal of this project is to develop a Python script that labels a set of clothing images based on their color and shape. 
The automated labeling enhances the search functionality on an online shop by providing accurate and updated product descriptions.

## Algorithms
### K-means Clustering
* **Purpose:** Label images based on their colors.
* **Type:** Unsupervised learning algorithm.
* **Functionality:** This algorithm autonomously creates clusters and assigns each pixel in the images to these clusters based on color similarities. It does not rely on predefined classes but instead identifies patterns and similarities within the data.
  
### K-Nearest Neighbors (KNN)
* **Purpose:** Categorize images according to their shapes.
* **Type:** Supervised learning algorithm.
* **Functionality:** This algorithm categorizes images into predefined shape classes by evaluating the proximity of each image to its neighboring instances in the feature space. It uses labeled training data to classify new images based on their shapes.

## Technologies
* **scipy:** For scientific and technical computing.
* **sklearn (scikit-learn):** For implementing machine learning algorithms.
* **numpy:** For numerical operations and array manipulation.
* **matplotlib:** For data visualization and plotting.
