__authors__ = ['1716921', '1718541', '1720318']
__group__ = '213'

#from sklearn.model_selection import train_test_split
from utils_data import read_dataset, read_extended_dataset, crop_images, visualize_retrieval
from Kmeans import KMeans, get_colors
from KNN import KNN
import numpy as np
import time
import matplotlib.pyplot as plt
from typing import Set


def retrival_by_colour(list_images, tags, question, percentage=None):
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


def retrival_by_shape(list_images, tags, question, percentage=None):
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


def retrival_combined(list_image, color_tags, shape_tags, colour_question, shape_question, colour_percentage=None,
                      shape_percentage=None):
    colour_images = retrival_by_colour(list_image, color_tags, colour_question, colour_percentage)
    shape_images = retrival_by_shape(list_image, shape_tags, shape_question, shape_percentage)
    combined_images = []
    for ci in colour_images:
        for si in shape_images:
            if np.array_equal(ci, si):
                combined_images.append(ci)
                break
    return combined_images


def plot_knn_accuracy_for_dataset(train_imgs, train_labels, test_imgs, test_labels, k_range, feature_method, downsample,
                                  downsample_factor=None, method_label=""):
    accuracies = []

    for k in k_range:
        knn = KNN(train_imgs, train_labels, feature_method=feature_method, downsample=downsample,
                  downsample_factor=downsample_factor)
        knn_predictions = knn.predict(test_imgs, k=k)
        accuracy = get_shape_accuracy(knn_predictions, test_labels)
        accuracies.append(accuracy)

        print(f"Method: {method_label}, K={k}: Accuracy = {accuracy:.2f}%")

    # Plotting
    plt.figure(figsize=(10, 5))
    plt.plot(k_range, accuracies, label=method_label, marker='o', color='blue')
    plt.xlabel('Number of Neighbors (k)')
    plt.ylabel('Accuracy (%)')
    plt.title(f'KNN Classification Accuracy vs. k ({method_label})')
    plt.legend()
    plt.grid(True)
    plt.show()


def visualize_knn_shape_predictions(imgs, labels_predicted, labels_actual, topN):
    fig = plt.figure(figsize=(12, topN // 4 * 3))  # Adjust size as needed
    for i in range(min(topN, len(imgs))):
        plt.subplot(topN // 4, 4, i + 1)
        plt.imshow(imgs[i])
        plt.title(f'Pred: {labels_predicted[i]}, Gt: {labels_actual[i]}')
        plt.axis('off')

    fig.subplots_adjust(left=0.05, right=0.95, top=0.9, bottom=0.1, hspace=0.4, wspace=0.3)
    plt.show()


def visualize_kmeans_color_predictions(imgs, accuracy_scores, predicted_labels, ground_truth_labels, topN):
    plt.figure(figsize=(10, (topN // 4) * 3))
    for i, (img, score, pred, gt) in enumerate(
            zip(imgs[:topN], accuracy_scores[:topN], predicted_labels[:topN], ground_truth_labels[:topN])):
        plt.subplot(topN // 4, 4, i + 1)
        plt.imshow(img)
        plt.title(f'F1: {score:.2f}', fontsize=10)  # Display the F1 score
        plt.xlabel(f'Pred: {pred}\nGT: {gt}', fontsize=8)  # Display predicted and ground truth labels
        plt.xticks([])  # Remove x-axis tick marks
        plt.yticks([])  # Remove y-axis tick marks
    plt.tight_layout()
    plt.subplots_adjust(hspace=0.5)
    plt.show()


def kmean_statistics(images, ground_truth_labels, Kmax, options=None):
    overall_accuracy_scores = []

    all_wcd_values = []
    all_inter_class_values = []
    all_fisher_values = []

    all_times = []
    all_iterations = []

    for k in range(2, Kmax + 1):
        wcd_values = []
        inter_class_dist_values = []
        fisher_coeff_values = []
        times = []
        iterations = []

        kmeans_predicted_labels = []

        for img in images:
            kmeans = KMeans(img, K=k, options=options)

            start = time.time()
            kmeans.fit()
            end = time.time()

            # Calculate statistics
            wcd = kmeans.withinClassDistance()
            wcd_values.append(wcd)

            inter_class_dist = kmeans.interClassDistance()
            inter_class_dist_values.append(inter_class_dist)

            fisher_coeff = kmeans.fisherDiscriminant()
            fisher_coeff_values.append(fisher_coeff)

            times.append(end - start)
            iterations.append(kmeans.num_iter)

            predicted_colors = get_colors(kmeans.centroids)
            kmeans_predicted_labels.append(set(predicted_colors))

        accuracy_scores = get_color_accuracy(kmeans_predicted_labels, ground_truth_labels)
        overall_f1_score = np.mean(
            [score for score in accuracy_scores])  # Average F1 score across all images for this k
        overall_accuracy_scores.append(overall_f1_score)

        print(f'K={k}: Average F1 score across all images: {overall_f1_score:.2f}')

        # Calculate average statistics
        avg_wcd = np.mean(wcd_values)
        avg_inter_class_distance = np.mean(inter_class_dist_values)
        avg_fisher_coeff = np.mean(fisher_coeff_values)

        avg_time = np.mean(times)
        avg_iterations = np.mean(iterations)

        all_wcd_values.append(avg_wcd)
        all_inter_class_values.append(avg_inter_class_distance)
        all_fisher_values.append(avg_fisher_coeff)
        all_times.append(avg_time)
        all_iterations.append(avg_iterations)

        print(f'K={k}: Avg WCD={avg_wcd:.4f}, Avg Time={avg_time:.4f}s, Avg Iterations={avg_iterations}')

    print("all_inter_class_values:", all_inter_class_values)
    print("all_fisher_values:", all_fisher_values)
    plt.figure(figsize=(10, 5))
    plt.plot(range(2, Kmax + 1), overall_accuracy_scores, marker='o', linestyle='-', color='blue')
    plt.xlabel('Number of Clusters (K)')
    plt.ylabel('Average F1 Score')
    plt.title('K-means Color Clustering Accuracy vs. Number of Clusters')
    plt.grid(True)
    plt.show()

    plt.figure(figsize=(18, 6))

    plt.subplot(1, 3, 1)
    plt.plot(range(2, Kmax + 1), all_wcd_values, marker='o')
    plt.xlabel('Number of Clusters (K)')
    plt.ylabel('Within-Class Distance (WCD)')
    plt.title('WCD vs K')

    plt.subplot(1, 3, 2)
    plt.plot(range(2, Kmax + 1), all_inter_class_values, marker='o')
    plt.xlabel('Number of Clusters (K)')
    plt.ylabel('Inter-Class Distance')
    plt.title('Inter-Class Distance vs K')

    plt.subplot(1, 3, 3)
    plt.plot(range(2, Kmax + 1), all_fisher_values, marker='o')
    plt.xlabel('Number of Clusters (K)')
    plt.ylabel('Fisher Coeff')
    plt.title('Fisher Coeff vs K')

    plt.grid(True)
    plt.subplots_adjust(left=0.05, right=0.95, wspace=0.3)
    plt.show()

    plt.figure(figsize=(18, 6))

    plt.subplot(1, 3, 1)
    plt.plot(range(2, Kmax + 1), all_times, marker='o', color='green')
    plt.xlabel('Number of Clusters (K)')
    plt.ylabel('Time (seconds)')
    plt.title('Computation Time vs K')

    plt.subplot(1, 3, 2)
    plt.plot(range(2, Kmax + 1), all_iterations, marker='x', color='red')
    plt.xlabel('Number of Clusters (K)')
    plt.ylabel('Iterations')
    plt.title('Iterations vs K')

    plt.subplots_adjust(left=0.05, right=0.95, wspace=0.3)
    plt.show()


def get_shape_accuracy(predicted_tags, ground_truth):
    """
    Calculate shape classification accuracy.
    Args:
        predicted_tags (np.array): Predicted shape labels by using KNN
        ground_truth (np.array): Ground truth shape labels

    Returns:
        float: Accuracy percentage
    """
    correct_count = np.sum(predicted_tags == ground_truth)
    total = len(ground_truth)
    accuracy = (correct_count / total) * 100  # convert to percentage
    return accuracy


# Precision: How many of the predicted tags are correct.
# Recall: How many of the ground-truth tags are predicted.
# F1 Score: Harmonic mean of Precision and Recall to give a single score
def compute_precision_recall(predicted_set: Set[str], ground_truth_set: Set[str]):
    """
    Compute Precision and Recall, f1 score for one image given two sets of tags, predicted and ground truth
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
        f1_score (float): F1 scores for all images.
    """
    precision_scores = []
    recall_scores = []
    f1_scores = []

    for pred, gt in zip(predicted_labels, ground_truth_labels):
        pred_set = set(pred)
        gt_set = set(gt)

        _, _, f1_score = compute_precision_recall(pred_set, gt_set)

        f1_scores.append(f1_score)

    return f1_scores


def threshold_accuracy(imgs, ground_truth_labels, options):
    accuracies = []

    thresholds = [i for i in range(10, 40, 5)]
    print("thresholds", thresholds)
    for i in range(40, 50, 5):
        good_k = 0
        kmeans_predicted_labels = []
        options['threshold'] = i
        for image in imgs:
            kmeans = KMeans(image, K=3, options=options)
            kmeans.find_bestK(10)
            kmeans.fit()
            predicted_colors = get_colors(kmeans.centroids)
            kmeans_predicted_labels.append(set(predicted_colors))

        for pred, gt in zip(kmeans_predicted_labels, ground_truth_labels):
            if len(pred) == len(gt):
                good_k += 1
        accuracies.append(good_k/len(imgs))
        # accuracy is computed only according to the number of colours put in the ground truth labels and not on their value

    plt.figure(figsize=(10, 5))
    plt.plot(thresholds, accuracies, marker='o', linestyle='-', color='blue')
    plt.xlabel('Percentage of the threshold')
    plt.ylabel('Average well determined K (%)')
    plt.title('K-means threshold Accuracy (FD) vs. percentage of the threshold')
    plt.grid(True)
    plt.show()


if __name__ == '__main__':

    # Load all the images and GT
    train_imgs, train_class_labels, train_color_labels, test_imgs, test_class_labels, \
        test_color_labels = read_dataset(root_folder='./images/', gt_json='./images/gt.json')

    # List with all the existent classes
    classes = list(set(list(train_class_labels) + list(test_class_labels)))

    # Load extended ground truth
    imgs, class_labels, color_labels, upper, lower, background = read_extended_dataset()
    cropped_images = crop_images(imgs, upper, lower)

    k_range = range(1, 11)
    options = {
        'km_init': 'first',
        'max_iter': 50,
        'tolerance': 1e-4,
        'threshold': 10,
        'fitting':'FD'
    }

    #########################################   KNN   #########################################

    #knn = KNN(train_imgs, train_class_labels, feature_method=1, downsample=False)
    ##train_class_labels ['Shorts' 'Heels' 'Shorts' ... 'Sandals' 'Shirts' 'Jeans']
    #knn_imgs_predictions = knn.predict(imgs, k=3)
    #knn_test_predictions = knn.predict(test_imgs, k=3)
    #
    ## Show predictions for the first 20 images: Prediction of KNN vs Ground Truth
    #visualize_knn_shape_predictions(test_imgs[:20], knn_test_predictions[:20], test_class_labels[:20], 20)
    #visualize_knn_shape_predictions(imgs[:20], knn_imgs_predictions[:20], class_labels[:20], 20)
    #
    #accuracy = get_shape_accuracy(knn_test_predictions, test_class_labels)
    #print("Shape classification accuracy for test set with k=3:", accuracy) #90.951821386604
    #accuracy = get_shape_accuracy(knn_imgs_predictions, class_labels)
    #print("Shape classification accuracy for imgs set with k=3:", accuracy) #93.33333333333333
    #
    ## Plot accuracy for normal flatten without downsample
    #plot_knn_accuracy_for_dataset(train_imgs, train_class_labels, test_imgs, test_class_labels, k_range, feature_method=1, downsample=False, method_label="Flatten Without Downsampling")
    #
    ## Plot accuracy for normal flatten with downsample
    #plot_knn_accuracy_for_dataset(train_imgs, train_class_labels, test_imgs, test_class_labels, k_range, feature_method=1, downsample=True, downsample_factor=2, method_label="Flatten With Downsampling")
    #
    ## Plot accuracy for custom features
    #plot_knn_accuracy_for_dataset(train_imgs, train_class_labels, test_imgs, test_class_labels, k_range, feature_method=2, downsample=False, method_label="Additional Custom Features [mean, var, upper, lower]")
    #
    #plot_knn_accuracy_for_dataset(train_imgs, train_class_labels, test_imgs, test_class_labels, k_range, feature_method=3, downsample=False, method_label="Additional Custom Features [hue_mean, saturation_mean, brightness_mean]")
    #

    #########################################   KMEANS    #########################################

    #kmeans_images = []
    #kmeans_predicted_labels = []
    #kmeans_predicted_cropped_labels = []
    #
    #for image in imgs:
    #    kmeans = KMeans(image, K=3, options=options)
    #    kmeans.fit()
    #    predicted_colors = get_colors(kmeans.centroids)
    #    kmeans_images.append(predicted_colors)
    #    kmeans_predicted_labels.append(set(predicted_colors))

    #for image in cropped_images:
    #    kmeans = KMeans(image, K=3, options=options)
    #    kmeans.fit()
    #    predicted_colors = get_colors(kmeans.centroids)
    #    kmeans_predicted_cropped_labels.append(set(predicted_colors))

    ground_truth_labels = []
    for gt_colors in color_labels:
        ground_truth_labels.append(set(gt_colors))

    #Accuracy scores = vectors of f1 scores with f1 score for each image comparing kmeans prediction with true label
    #Example accuracy scores [0.8, 0.5, 0, 0.4, 0, 0, 0.4, 0.5, 0.5, 0, 0.4, 0.3333333333333333, 0.5, 0.8 , ...]
    #accuracy_scores = get_color_accuracy(kmeans_predicted_labels, ground_truth_labels)
    #accuracy_scores_cropped = get_color_accuracy(kmeans_predicted_cropped_labels, ground_truth_labels)

    # Visualization showing for each image: F1 score, predicted colors, label colors
    #visualize_kmeans_color_predictions(cropped_images, accuracy_scores_cropped, kmeans_predicted_cropped_labels, ground_truth_labels, 20)
    #visualize_kmeans_color_predictions(imgs, accuracy_scores, kmeans_predicted_labels, ground_truth_labels, 20)

    #visualize_kmeans_color_predictions(cropped_images[20:], accuracy_scores_cropped[20:], kmeans_predicted_cropped_labels[20:], ground_truth_labels[20:], 20)
    #visualize_kmeans_color_predictions(imgs[20:, :, :, :], accuracy_scores[20:], kmeans_predicted_labels[20:],
    #                                   ground_truth_labels[20:], 20)

    #TO PLOT FOR REPORT COULD CHANGE CROPPED_IMAGES TO IMAGES AND KMAX TO OTHER 
    #K-means Color Clustering Accuracy, Within Class Distance, nr of iterations, time (needed to converge) vs K
    #print("K-Mean Statistics Analysis: ")
    #kmean_statistics(imgs, ground_truth_labels, Kmax=10, options=options)

    threshold_accuracy(imgs, ground_truth_labels, options)

    ###############################QUALITATIVE ANALYSIS FUNCTIONS##################################

    ## Visualise images
    #visualize_retrieval(imgs[:60, :, :, :], 60, title="First 60 Images")
    #visualize_retrieval(imgs[60:120, :, :, :], 60, title="Next 60 Images")
    #visualize_retrieval(imgs[120:180, :, :, :], 60, title="Third 60 Images")
    #
    ##Retrieve and visualize images by shape for "Shorts"
    #images1 = retrival_by_shape(imgs, knn_imgs_predictions, "Shorts")
    #visualize_retrieval(images1, 20, title="Images of Shorts")
    #
    ## Retrieve and visualize images by shape for "Dresses"
    #images2 = retrival_by_shape(imgs, knn_imgs_predictions, "Dresses")
    #visualize_retrieval(images2, 20, title="Images of Dresses")
    #
    ## Retrieve and visualize images by color "White"
    #colour1 = retrival_by_colour(imgs, kmeans_images, "White")
    #visualize_retrieval(colour1, 50, title="Images with White Color")
    #
    ## Retrieve and visualize combined images with specific color and shape
    #combined = retrival_combined(imgs, kmeans_images, class_labels, "Blue", "Shorts")
    #visualize_retrieval(combined, 20, title="Color: Blue, Shape: Shorts")
