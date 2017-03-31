import os
import numpy as np
import cv2
import params


def imreads(path):
    """
    This reads all the images in a given folder and returns the results
    """
    images_path = [os.path.join(path, f) for f in os.listdir(path)]
    images = []
    for image_path in images_path:
        img = cv2.imread(image_path)
        images.append(img)
    return images


def stack_array(arr):
    stacked_arr = np.array([])
    for item in arr:
        if len(stacked_arr) == 0:
            stacked_arr = np.array(item)
        else:
            stacked_arr = np.vstack((stacked_arr, item))
    return stacked_arr


def get_descriptors(img):
    # returns descriptors of an image
    return params.DETECTOR.detectAndCompute(img, None)[1]


def get_class_code(class_name):
    return params.CLASS_NAMES.get(class_name, 0)


class ImageData(object):
    def __init__(self, img):
        self.img = img
        self.class_name = ""
        self.response = None
        self.descriptors = np.array([])

    def set_class(self, class_name):
        self.class_name = class_name
        self.response = get_class_code(self.class_name)

    def compute_descriptors(self):
        self.descriptors = get_descriptors(self.img)

    def generate_bow_hist(self, dictionary):
        self.features = np.zeros((len(dictionary), 1))
        # FLANN matcher needs descriptors to be type32
        matches = params.MATCHER.match(np.float32(self.descriptors), dictionary)
        for match in matches:
            # Get which visual word this descriptor matches in the dictionary
            # match.trainIdx is the visual_word
            # Increase count for this visual word in histogram
            self.features[match.trainIdx] += 1


def get_samples(imgs_data):
    samples = stack_array([[img_data.features] for img_data in imgs_data])
    return np.float32(samples)


def get_responses(imgs_data):
    responses = [img_data.response for img_data in imgs_data]
    return np.int32(responses)
