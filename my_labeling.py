__authors__ = ['1716921', '1718541', '1720318']
__group__ = '213'

from utils_data import read_dataset, read_extended_dataset, crop_images, visualize_retrieval
from Kmeans import KMeans
import numpy as np


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
        visualize_retrieval(colour_images, 20)
        shape_images = Retrival_by_shape(list_image, shape_tags, shape_question, shape_percentage)
        visualize_retrieval(shape_images, 20)
        combined_images = []
        for ci in colour_images:
            for si in shape_images:
                if np.array_equal(ci, si):
                    combined_images.append(ci)
                    break
        return combined_images

    # test of the functions
    images1 = Retrival_by_shape(imgs, class_labels,"Shorts")
    images2 = Retrival_by_shape(train_imgs, train_class_labels, "Dresses")
    visualize_retrieval(images1, 20)
    visualize_retrieval(images2, 100)

    colour1 = Retrival_by_colour(imgs, color_labels, "White")
    visualize_retrieval(colour1, 50)

    combined = Retrival_combined(imgs, color_labels, class_labels, "Blue", "Shorts")
    visualize_retrieval(combined, 20)