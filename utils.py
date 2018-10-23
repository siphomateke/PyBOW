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
import random

################################################################################
# global flags to facilitate output of additional info per stage/function

show_additional_process_information = True;
show_images_as_they_are_loaded = True;
show_images_as_they_are_sampled = True;

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
# used in the multi-scale image pyramid approach

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

def read_all_images(path):
    images_path = [os.path.join(path, f) for f in os.listdir(path)]
    images = []
    for image_path in images_path:
        img = cv2.imread(image_path)

        if show_additional_process_information:
            print("loading file - ", image_path);

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

def get_class_number(class_name):
    return params.DATA_CLASS_NAMES.get(class_name, 0)

def get_class_name(class_code):
    for name, code in params.DATA_CLASS_NAMES.items():
        if code == class_code:
            return name

################################################################################

# image data class object that contains the images, descriptors and bag of word
# histograms

class ImageData(object):
    def __init__(self, img):
        self.img = img
        self.class_name = ""
        self.class_number = None
        self.descriptors = np.array([])

    def set_class(self, class_name):
        self.class_name = class_name
        self.class_number = get_class_number(self.class_name)
        if show_additional_process_information:
            print("class name : ", class_name, " - ", self.class_number);

    def compute_descriptors(self):

        # generate the feature descriptors for a given image

        self.descriptors = get_descriptors(self.img)

        if self.descriptors is None:
            self.descriptors = np.array([])

        if show_additional_process_information:
            print("# feature descriptors computed - ", len(self.descriptors));

    def generate_bow_hist(self, dictionary):
        self.bow_histogram = np.zeros((len(dictionary), 1))

        # generate the bow histogram of feature occurance from descriptors

        # FLANN matcher needs descriptors to be type32
        matches = params.MATCHER.match(np.float32(self.descriptors), dictionary)
        for match in matches:
            # Get which visual word this descriptor matches in the dictionary
            # match.trainIdx is the visual_word
            # Increase count for this visual word in histogram (known as hard assignment)
            self.bow_histogram[match.trainIdx] += 1

        # Important! - normalize the histogram to L1 to remove bias for number
        # of descriptors per image or class (could use L2?)

        self.bow_histogram = cv2.normalize(self.bow_histogram, None, alpha=1, beta=0, norm_type=cv2.NORM_L1);

################################################################################

# generates a set of random sample patches from a given image of a specified size

def generate_patches(img, sample_patches_to_generate=0, patch_size=(64,128)):

    patches = [];

    # if no patches specifed just return original image

    if (sample_patches_to_generate == 0):
        return [img];

    # otherwise generate N sub patches

    else:

        # get all heights and widths

        img_height, img_width, _ = img.shape;
        patch_height = patch_size[1];
        patch_width = patch_size[0];

        # iterate to find up to N patches (0 -> N-1)

        for patch_count in range(sample_patches_to_generate):

            # randomly select a patch, ensuring we stay inside the image

            patch_start_h =  random.randint(0, (img_height - patch_height));
            patch_start_w =  random.randint(0, (img_width - patch_width));

            # add this patch to the list of patches

            patch = img[patch_start_h:patch_start_h + patch_height, patch_start_w:patch_start_w + patch_width]

            if (show_images_as_they_are_sampled):
                cv2.imshow("patch", patch);
                cv2.waitKey(5);

            patches.insert(patch_count, patch);

        return patches;

################################################################################

# add images from a specified path to the dataset, adding the appropriate class/type name
# and optionally adding up to N samples of a specified size

def load_image_path(path, class_name, imgs_data, samples=0, patch_size=(64,128)):

    # read all images at location

    imgs = read_all_images(path)

    img_count = len(imgs_data)
    for img in imgs:

        if (show_images_as_they_are_loaded):
            cv2.imshow("example", img);
            cv2.waitKey(5);

        # generate up to N sample patches for each sample image
        # if zero samples is specified then generate_patches just returns
        # the original image (unchanged, unsampled) as [img]

        for img_patch in generate_patches(img, samples, patch_size):

            if show_additional_process_information:
                print("path: ", path, "class_name: ", class_name, "patch: ", img_count)

            # add each image patch to the data set

            img_data = ImageData(img_patch)
            img_data.set_class(class_name)
            imgs_data.insert(img_count, img_data)
            img_count += 1

    return imgs_data

################################################################################

# load image data from specified paths

def load_images(paths, class_names, sample_set_sizes, patch_size=(64,128)):
    imgs_data = []  # type: list[ImageData]

    # for each specified path and corresponding class_name and required number
    # of samples - add them to the data set

    for path, class_name, sample_count in zip(paths, class_names, sample_set_sizes):
        load_image_path(path, class_name, imgs_data, sample_count, patch_size)

    return imgs_data

################################################################################

# return the global set of bow histograms for the data set of images

def get_bow_histograms(imgs_data):

    samples = stack_array([[img_data.bow_histogram] for img_data in imgs_data])
    return np.float32(samples)

################################################################################

# return global the set of numerical class labels for the data set of images

def get_class_labels(imgs_data):
    class_labels = [img_data.class_number for img_data in imgs_data]
    return np.int32(class_labels)

################################################################################
