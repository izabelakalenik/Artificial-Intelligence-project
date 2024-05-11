__authors__ = ['1716921', '1718541', '1720318']
__group__ = '213'

from utils_data import read_dataset, read_extended_dataset, crop_images, visualize_retrieval
from Kmeans import KMeans, get_colors
from KNN import KNN
import numpy as np
import time
import matplotlib.pyplot as plt
from typing import Set

def Kmean_statistics(images, Kmax, options=None):    
    all_wcd_values = []
    all_times = []
    all_iterations = []

    for k in range(2, Kmax + 1):
        wcd_values = []
        times = []
        iterations = []
        for img in images:
            kmeans = KMeans(img, K=k, options=options)

            start = time.time()
            kmeans.fit()
            end = time.time()

            # Calculate statistics
            wcd = kmeans.withinClassDistance()
            wcd_values.append(wcd)
            times.append(end - start)
            iterations.append(kmeans.num_iter)

        # Calculate average statistics
        avg_wcd = np.mean(wcd_values)
        avg_time = np.mean(times)
        avg_iterations = np.mean(iterations)

        all_wcd_values.append(avg_wcd)
        all_times.append(avg_time)
        all_iterations.append(avg_iterations)

        print(f'K={k}: Avg WCD={avg_wcd:.4f}, Avg Time={avg_time:.4f}s, Avg Iterations={avg_iterations}')

    # Maybe we ll change the plotting i m not 100% sure about it
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.plot(range(2, Kmax + 1), all_wcd_values, marker='o')
    plt.xlabel('Number of Clusters (K)')
    plt.ylabel('Within-Class Distance (WCD)')
    plt.title('WCD vs K')

    plt.subplot(1, 2, 2)
    plt.plot(range(2, Kmax + 1), all_times, marker='o', label='Time')
    plt.plot(range(2, Kmax + 1), all_iterations, marker='x', label='Iterations')
    plt.xlabel('Number of Clusters (K)')
    plt.legend()
    plt.title('Time and Iterations vs K')

    plt.show()

def Get_shape_accuracy(predicted_tags, ground_truth):
    """
    Calculate shape classification accuracy.
    Args:
        predicted_tags (np.array): Predicted shape labels by using KNN
        ground_truth (np.array): Ground truth shape labels

    Returns:
        float: Accuracy percentage
    """
    correct_count = np.sum(np.array(predicted_tags) == np.array(ground_truth))
    accuracy = correct_count / len(ground_truth) * 100
    return accuracy

# I leave them just for now
# Measuring the similarity between these sets - Precision and Recall:
# Precision: How many of the predicted tags are correct.
# Recall: How many of the ground-truth tags are predicted.
# F1 Score: Harmonic mean of Precision and Recall to give a single score
def compute_precision_recall(predicted_set: Set[str], ground_truth_set: Set[str]):
    """
    Compute Precision and Recall given two sets of tags.
    Example: ['Brown', 'Grey', 'Orange', 'White'] versus ['White', 'Orange']
    Args:
        predicted_set (set): Predicted color tags.
        ground_truth_set (set): Ground-truth color tags.

    Returns:
        precision (float): Precision between 0 and 1.
        recall (float): Recall between 0 and 1.
        f1_score (float): F1 score between 0 and 1.
    """
    intersection = predicted_set.intersection(ground_truth_set)
    true_positives = len(intersection)

    precision = true_positives / len(predicted_set) if len(predicted_set) > 0 else 0
    recall = true_positives / len(ground_truth_set) if len(ground_truth_set) > 0 else 0

    if precision + recall == 0:
        f1_score = 0
    else:
        f1_score = 2 * (precision * recall) / (precision + recall)

    return precision, recall, f1_score


def get_color_accuracy(predicted_labels, ground_truth_labels):
    """
    Calculates the color tag accuracy using Precision, Recall, and F1 scores.
    Args:
        predicted_labels (np.array): Labels obtained from KMeans clustering.
        ground_truth_labels (np.array): Ground-truth color labels for each image.

    Returns:
        precision (float): Average precision across all images.
        recall (float): Average recall across all images.
        f1_score (float): Average F1 score across all images.
    """
    precision_scores = []
    recall_scores = []
    f1_scores = []

    for pred, gt in zip(predicted_labels, ground_truth_labels):
        pred_set = set(pred)
        gt_set = set(gt)

        precision, recall, f1_score = compute_precision_recall(pred_set, gt_set)

        precision_scores.append(precision)
        recall_scores.append(recall)
        f1_scores.append(f1_score)

    return np.mean(precision_scores), np.mean(recall_scores), np.mean(f1_scores)


if __name__ == '__main__':

    # Load all the images and GT
    train_imgs, train_class_labels, train_color_labels, test_imgs, test_class_labels, \
        test_color_labels = read_dataset(root_folder='./images/', gt_json='./images/gt.json')

    # List with all the existent classes
    classes = list(set(list(train_class_labels) + list(test_class_labels)))

    # Load extended ground truth
    imgs, class_labels, color_labels, upper, lower, background = read_extended_dataset()
    cropped_images = crop_images(imgs, upper, lower)

    # You can start coding your functions here
    def Retrival_by_colour(list_images, tags, question, percentage=None):
        possible_images = []
        list_percentages = []
        question = [question]
        for i in range(len(tags)):
            check = all(colour in tags[i] for colour in question)
            if check:
                possible_images.append(list_images[i])
                if percentage is not None:
                    list_percentages.append(np.mean([percentage[i][j] for j in question]))
        if percentage is not None:
            possible_images = [images for percentage, images in sorted(zip(list_percentages, possible_images))]
        return possible_images

    def Retrival_by_shape(list_images, tags, question, percentage=None):
        possible_images = []
        list_percentages = []
        for i in range(len(tags)):
            if question in tags[i]:
                possible_images.append(list_images[i])
                if percentage is not None:
                    list_percentages.append(percentage[i][question])
        if percentage is not None:
            possible_images = [images for percentage, images in sorted(zip(list_percentages, possible_images))]
        return possible_images


    def Retrival_combined(list_image, color_tags, shape_tags, colour_question, shape_question, colour_percentage=None, shape_percentage=None):
        colour_images = Retrival_by_colour(list_image, color_tags, colour_question, colour_percentage)
        shape_images = Retrival_by_shape(list_image, shape_tags, shape_question, shape_percentage)
        combined_images = []
        for ci in colour_images:
            for si in shape_images:
                if np.array_equal(ci, si):
                    combined_images.append(ci)
                    break
        return combined_images

    # Run kmeans and knn
    options = {
        'km_init': 'random',
        'max_iter': 10,
        'tolerance': 1e-4
    }

    #kmeans_predictions = []
    #for img in cropped_images:
        #kmeans = KMeans(img, K=5, options=options)
        #kmeans.fit()
        #predicted_colors = get_colors(kmeans.centroids)
        #kmeans_predictions.append(predicted_colors)

    knn = KNN(train_imgs, train_class_labels)
    #knn_predictions = knn.predict(test_imgs, k=3)

    kmeans_images = []
    for image in imgs:
        kmeans = KMeans(image, K=5, options=options)
        kmeans.fit()
        predicted_colors = get_colors(kmeans.centroids)
        kmeans_images.append(predicted_colors)

    knn_qualitative = knn.predict(imgs, k=3)

    # test of the functions
    visualize_retrieval(imgs[:60, :, :, :], 60)
    visualize_retrieval(imgs[60:120, :, :, :], 60)
    visualize_retrieval(imgs[120:180, :, :, :], 60)

    images1 = Retrival_by_shape(imgs, knn_qualitative, "Shorts")
    images2 = Retrival_by_shape(imgs, knn_qualitative, "Dresses")
    visualize_retrieval(images1, 20)
    visualize_retrieval(images2, 20)

    colour1 = Retrival_by_colour(imgs, kmeans_images, "White")
    visualize_retrieval(colour1, 50)

    combined = Retrival_combined(imgs, kmeans_images, class_labels, "Blue", "Shorts")
    visualize_retrieval(combined, 20)

    #print("K-Mean Statistics Analysis")
    #Kmean_statistics(imgs, Kmax=10, options=options)

    #print("\nK-Means Color Prediction")
    #color_precision, color_recall, color_f1 = get_color_accuracy(kmeans_predictions, color_labels)
    #print(f"Avarage Color Accuracy - Precision for all images: {color_precision:.2f}, Recall: {color_recall:.2f}, F1 Score: {color_f1:.2f}")

    #print("\nKNN Shape Prediction")
    #shape_accuracy = Get_shape_accuracy(knn_predictions, test_class_labels)
    #there is a problem with the shapes -  i will look later - but i just wanted to have it uploaded
    #print(f"Shape Classification Accuracy: {shape_accuracy:.2f}%")