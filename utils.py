################################################################################

# functionality: utility functions for all detection algorithms

# This version: (c) 2018 Toby Breckon, Dept. Computer Science, Durham University, UK
# License: MIT License

# Origin acknowledgements: forked from https://github.com/nextgensparx/PyBOW

################################################################################

import os
import numpy as np
import cv2
import params

################################################################################
# global flag to facilitate output of additional info per stage/function

show_additional_process_information = True;

################################################################################

# timing information - for training

def get_elapsed_time(start):
    """ Helper function for timing code execution"""
    return (cv2.getTickCount() - start) / cv2.getTickFrequency()


def format_time(time):
    time_str = ""
    if time < 60.0:
        time_str = "{}s".format(round(time, 1))
    elif time > 60.0:
        minutes = time / 60.0
        time_str = "{}m : {}s".format(int(minutes), round(time % 60, 2))
    return time_str


def print_duration(start):
    time = get_elapsed_time(start)
    print(("Took {}".format(format_time(time))))

################################################################################

# re-size an image with respect to its aspect ratio if needed.
# Used in the construction of the supplied basic sliding window approach

def resize_img(img, width=-1, height=-1):
    if height == -1 and width == -1:
        raise TypeError("Invalid arguments. Width or height must be provided.")
    h = img.shape[0]
    w = img.shape[1]
    if height == -1:
        aspect_ratio = float(w) / h
        new_height = int(width / aspect_ratio)
        return cv2.resize(img, (width, new_height))
    elif width == -1:
        aspect_ratio = h / float(w)
        new_width = int(height / aspect_ratio)
        return cv2.resize(img, (new_width, height))

################################################################################

# reads all the images in a given folder path and returns the results

# for obvious reasons this will break with a very large dataset as you will run
# out of memory - so an alternative approach may be required in that case

def imreads(path):
    images_path = [os.path.join(path, f) for f in os.listdir(path)]
    images = []
    for image_path in images_path:
        img = cv2.imread(image_path)

        if show_additional_process_information:
            print("loading file - ", image_path);

        # to debug image loading uncomment lines below
        # cv2.imshow("loaded", img)
        # cv2.waitKey(200)

        images.append(img)
    return images

################################################################################

# stack array of items as basic Pyton data manipulation

def stack_array(arr):
    stacked_arr = np.array([])
    for item in arr:
        # Only stack if it is not empty
        if len(item) > 0:
            if len(stacked_arr) == 0:
                stacked_arr = np.array(item)
            else:
                stacked_arr = np.vstack((stacked_arr, item))
    return stacked_arr

################################################################################

# returns descriptors of an image - used in training and testing

def get_descriptors(img):
    return params.DETECTOR.detectAndCompute(img, None)[1]

################################################################################

# transform between class numbers (i.e. codes) - {0,1,2, ...N} and
# names {dog,cat cow, ...} - used in training and testing

def get_class_code(class_name):
    return params.DATA_CLASS_NAMES.get(class_name, 0)

def get_class_name(class_code):
    for name, code in params.DATA_CLASS_NAMES.items():
        if code == class_code:
            return name

################################################################################

# image data class object used in training and batch testing only

class ImageData(object):
    def __init__(self, img):
        self.img = img
        self.class_name = ""
        self.response = None
        self.descriptors = np.array([])

    def set_class(self, class_name):
        self.class_name = class_name
        self.response = get_class_code(self.class_name)
        if show_additional_process_information:
            print("class name : ", class_name, " - ", self.response);

    def compute_descriptors(self):

        # generate the feature descriptors for a given image

        self.descriptors = get_descriptors(self.img)

        if show_additional_process_information:
            print("# feature descripors computed - ", len(self.descriptors));

        if self.descriptors is None:
            self.descriptors = np.array([])

    def generate_bow_hist(self, dictionary):
        self.features = np.zeros((len(dictionary), 1))

        # generate the bow histogram of feature occurance from descriptors

        # FLANN matcher needs descriptors to be type32
        matches = params.MATCHER.match(np.float32(self.descriptors), dictionary)
        for match in matches:
            # Get which visual word this descriptor matches in the dictionary
            # match.trainIdx is the visual_word
            # Increase count for this visual word in histogram
            self.features[match.trainIdx] += 1

################################################################################

# add images from a specified path to the dataset

def add_to_imgs_data(path, class_name, imgs_data):

    # read all images at location

    imgs = imreads(path)

    img_count = len(imgs_data)
    for img in imgs:

        # add each image to the data set

        img_data = ImageData(img)
        img_data.set_class(class_name)
        imgs_data.insert(img_count, img_data)
        img_count += 1

    return imgs_data

################################################################################

# load image data from specified paths

def get_imgs_data(paths, class_names, dictionary=None):
    imgs_data = []  # type: list[ImageData]

    for path, class_name in zip(paths, class_names):
        add_to_imgs_data(path, class_name, imgs_data)

    [img_data.compute_descriptors() for img_data in imgs_data]
    if dictionary is not None:
        [img_data.generate_bow_hist(dictionary) for img_data in imgs_data]

    return imgs_data

################################################################################

# return the set of bow dictionary encoded features for the set of data set images

def get_samples(imgs_data):

    # Important! Normalize histograms to remove bias for number of descriptors
    # per image or class

    norm_features = [cv2.normalize(img_data.features, None, 0, len(img_data.features), cv2.NORM_MINMAX) for img_data in
                     imgs_data]

    samples = stack_array([[feature] for feature in norm_features])

    if show_additional_process_information:
        print("# samples encoded to bow histograms - ", len(norm_features));

    return np.float32(samples)

################################################################################

# return the set of numerical class labelss

def get_responses(imgs_data):
    responses = [img_data.response for img_data in imgs_data]
    return np.int32(responses)

################################################################################
