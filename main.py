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


def get_descriptors(img, detector):
    # returns descriptors of an image
    return detector.detectAndCompute(img, None)[1]


class ImageData(object):
    def __init__(self):
        global dictionary_size
        self.class_name = ""
        self.response = None
        self.descriptors = np.array([])
        # self.features = np.zeros((dictionary_size, 1), np.float32)
        self.features = np.zeros((dictionary_size, 1))

    def generate_bow_hist(self, matcher, dictionary):
        # flann matcher needs descriptors to be type32
        matches = matcher.match(np.float32(self.descriptors), dictionary)
        for match in matches:
            # Get which visual word this descriptor matches in the dictionary
            # match.trainIdx is the visual_word
            # Increase count for this visual word in histogram
            self.features[match.trainIdx] += 1


imgs_path = "train\pos"  # directory of images

dictionary_size = 512
# Loading images
imgs_data = []  # type: list[ImageData]
# imreads returns a list of all images in that directory
imgs = imreads(imgs_path)
for i in xrange(len(imgs)):
    imgs_data.insert(i, ImageData())
    # imgs_data.insert(i, np.zeros((dictionary_size, 1)))

detector = cv2.AKAZE_create()

FLANN_INDEX_KDTREE = 0
index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
search_params = dict(checks=50)
matcher = cv2.FlannBasedMatcher(index_params, search_params)

# Extracting descriptors
desc = np.array([])
for i in xrange(len(imgs)):
    img = imgs[i]
    descriptors = get_descriptors(img, detector)
    imgs_data[i].descriptors = descriptors
    if len(desc) == 0:
        desc = np.array(descriptors)
    else:
        desc = np.vstack((desc, descriptors))
# important, cv2.kmeans only accepts type32 descriptors
desc = np.float32(desc)

# Clustering
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 0.01)
flags = cv2.KMEANS_PP_CENTERS
# desc is a type32 numpy array of vstacked descriptors
compactness, labels, dictionary = cv2.kmeans(desc, dictionary_size, None, criteria, 1, flags)

# Generate histograms using matcher
[img_data.generate_bow_hist(matcher, dictionary) for img_data in imgs_data]

ax = plt.subplot(111)
ax.set_title("FLANN Histogram")
ax.set_xlabel("Visual words")
ax.set_ylabel("Frequency")
ax.plot(imgs_data[0].features.ravel())

plt.show()
