import os

import cv2
import numpy as np
from matplotlib import pyplot as plt


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


def get_descriptors(img, detector):
    # returns descriptors of an image
    return detector.detectAndCompute(img, None)[1]


def get_class_code(class_name):
    return {
        "pos": 0,
        "neg": 1
    }.get(class_name, 0)


class ImageData(object):
    def __init__(self, img):
        self.img = img
        self.class_name = ""
        self.response = None
        self.descriptors = np.array([])
        # self.features = np.zeros((dictionary_size, 1), np.float32)

    def set_class(self, class_name):
        self.class_name = class_name
        self.response = get_class_code(self.class_name)

    def compute_descriptors(self, detector):
        self.descriptors = get_descriptors(self.img, detector)

    def generate_bow_hist(self, matcher, dictionary):
        self.features = np.zeros((len(dictionary), 1))
        # flann matcher needs descriptors to be type32
        matches = matcher.match(np.float32(self.descriptors), dictionary)
        for match in matches:
            # Get which visual word this descriptor matches in the dictionary
            # match.trainIdx is the visual_word
            # Increase count for this visual word in histogram
            self.features[match.trainIdx] += 1


def get_samples(imgs_data):
    """samples = np.array([])
    for img_data in imgs_data:
        if len(samples) == 0:
            samples = np.array([img_data.features])
        else:
            samples = np.vstack((samples, [img_data.features]))"""
    samples = stack_array([[img_data.features] for img_data in imgs_data])
    return np.float32(samples)


def generate_dictionary(imgs_data, dictionary_size):
    # Extracting descriptors
    """desc = np.array([])
    for img_data in imgs_data:
        descriptors = img_data.descriptors
        if len(desc) == 0:
            desc = np.array(descriptors)
        else:
            desc = np.vstack((desc, descriptors))"""
    desc = stack_array([img_data.descriptors for img_data in imgs_data])
    # important, cv2.kmeans only accepts type32 descriptors
    desc = np.float32(desc)

    # Clustering
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 0.01)
    flags = cv2.KMEANS_PP_CENTERS
    # desc is a type32 numpy array of vstacked descriptors
    compactness, labels, dictionary = cv2.kmeans(desc, dictionary_size, None, criteria, 1, flags)

    return dictionary


def get_dictionary(train, imgs_data=None, dictionary_size=None):
    if train:
        dictionary = generate_dictionary(imgs_data, dictionary_size)
        np.save("ml/dictionary.npy", dictionary)
        return dictionary
    else:
        return np.load("ml/dictionary.npy")


def get_test_samples(pos, neg, dictionary):
    imgs_data = []  # type: list[ImageData]

    pos_imgs_path = pos
    neg_imgs_path = neg

    # imreads returns a list of all images in that directory
    pos_imgs = imreads(pos_imgs_path)
    neg_imgs = imreads(neg_imgs_path)

    img_count = 0
    for img in pos_imgs:
        img_data = ImageData(img)
        img_data.set_class("pos")
        imgs_data.insert(img_count, img_data)
        img_count += 1

    for img in neg_imgs:
        img_data = ImageData(img)
        img_data.set_class("neg")
        imgs_data.insert(img_count, img_data)
        img_count += 1

    [img_data.compute_descriptors(detector) for img_data in imgs_data]
    [img_data.generate_bow_hist(matcher, dictionary) for img_data in imgs_data]

    samples = get_samples(imgs_data)
    responses = np.int32([img_data.response for img_data in imgs_data])

    return samples, responses


dictionary_size = 512
# Loading images
imgs_data = []  # type: list[ImageData]

pos_imgs_path = "train/pos"
neg_imgs_path = "train/neg"

print("Loading images...")

# imreads returns a list of all images in that directory
pos_imgs = imreads(pos_imgs_path)
neg_imgs = imreads(neg_imgs_path)

img_count = 0
for img in pos_imgs:
    img_data = ImageData(img)
    img_data.set_class("pos")
    imgs_data.insert(img_count, img_data)
    img_count += 1

for img in neg_imgs:
    img_data = ImageData(img)
    img_data.set_class("neg")
    imgs_data.insert(img_count, img_data)
    img_count += 1

detector = cv2.AKAZE_create()

FLANN_INDEX_KDTREE = 0
index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
search_params = dict(checks=50)
matcher = cv2.FlannBasedMatcher(index_params, search_params)

print("Computing descriptors...")
[img_data.compute_descriptors(detector) for img_data in imgs_data]

print("Clustering...")
dictionary = get_dictionary(False, imgs_data, dictionary_size)

print("Generating histograms...")
[img_data.generate_bow_hist(matcher, dictionary) for img_data in imgs_data]

print("Training SVM...")

# Begin training SVM
svm = cv2.ml.SVM_create()
svm.setType(cv2.ml.SVM_C_SVC)
svm.setKernel(cv2.ml.SVM_LINEAR)

# Compile samples
samples = get_samples(imgs_data)
responses = np.int32([img_data.response for img_data in imgs_data])

svm.setTermCriteria((cv2.TERM_CRITERIA_COUNT, 1000, 1.e-06))
svm.train(samples, cv2.ml.ROW_SAMPLE, responses)

# samples = np.float32([imgs_data[0].features])
print svm.predict(samples)[1].ravel()

test_samples, test_responses = get_test_samples("test/pos", "test/neg", dictionary)

output = svm.predict(test_samples)[1].ravel()
print output

error = ((np.absolute(test_responses.ravel() - output).sum()) / float(output.shape[0])) * 100
print "Error in test data: {}%".format(error)
