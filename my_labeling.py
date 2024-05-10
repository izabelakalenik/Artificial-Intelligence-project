__authors__ = ['1716921', '1718541', '1720318']
__group__ = '213'

from utils_data import read_dataset, read_extended_dataset, crop_images
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
        for i in range(np.shape(tags)[0]):
            if list(question) in tags[i, :]:
                possible_images.append(list_images[i])
                if percentage is not None:
                    list_percentages.append(np.mean([percentage[i][j] for j in list(question)]))
        if percentage is not None:
            possible_images = [images for percentage, images in sorted(zip(list_percentages, possible_images))]
        return possible_images

    def Retrival_by_shape(list_images, tags, question, percentage=None):
        possible_images = []
        list_percentages = []
        for i in range(np.shape(tags)[0]):
            if question in tags[i, :]:
                possible_images.append(list_images[i])
                if percentage is not None:
                    list_percentages.append(percentage[i][question])
        if percentage is not None:
            possible_images = [images for percentage, images in sorted(zip(list_percentages, possible_images))]
        return possible_images


