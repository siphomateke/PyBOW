import cv2
import numpy as np

import imutils


def get_elapsed_time(start):
    """ Helper function for timing code execution"""
    return (cv2.getTickCount() - start) / cv2.getTickFrequency()


def get_detector_matcher():
    detector = cv2.AKAZE_create()
    matcher = cv2.FlannBasedMatcher_create()
    return detector, matcher


def get_descriptors(img, detector):
    return detector.detectAndCompute(img, None)[1]


def get_batch_descriptors(img_files, detector):
    desc = np.array([])
    descriptor_src_img = []
    for i in xrange(len(img_files)):
        img = img_files[i].img
        descriptors = get_descriptors(img, detector)
        if len(desc) == 0:
            desc = np.array(descriptors)
        else:
            desc = np.vstack((desc, descriptors))
        # Keep track of which image a descriptor belongs to
        for j in range(len(descriptors)):
            descriptor_src_img.append(i)
    # important, cv2.kmeans only accepts type32 descriptors
    desc = np.float32(desc)
    return desc, descriptor_src_img


class ImageData(object):
    """ Used to store an images bag of words """
    def __init__(self, size, class_name):
        self.bow_features = np.zeros((size, 1))
        self.class_name = class_name


def get_bow_features(matcher, descriptors, dictionary_size):
    output = np.zeros((dictionary_size, 1), np.float32)
    # flan matcher needs descriptors to be type32
    matches = matcher.match(np.float32(descriptors))
    for match in matches:
        visual_word = match.trainIdx
        output[visual_word] += 1
    return output


dictionary_size = 512
imgs_path = "train\pos"
# test_img = "[name of image to compare descriptors for]"
test_img = "Achaea lienardi, Noctuidae, Catocalinae M-2006-084.jpg"

print("Loading images...")
imgs_data = []
imgs = []
img_files = imutils.imreads(imgs_path)
for img_file in img_files:
    imgs_data.append(ImageData(dictionary_size, 0))
    imgs.append(img_file)

# Resizes images to a fixed width
imutils.resize_img_files(imgs, 320)
# Converts images to their grayscale equivalent
[img_file.to_gray() for img_file in imgs]

print("Extracting descriptors...")
start = cv2.getTickCount()
detector, matcher = get_detector_matcher()
# descriptor_imgs is a list which says what the id of each descriptors image is in all_imgs
desc, desc_src_img = get_batch_descriptors(imgs, detector)
print("Time elapsed: {}s".format(get_elapsed_time(start)))

print("Clustering...")
start = cv2.getTickCount()
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 0.01)
flags = cv2.KMEANS_PP_CENTERS
compactness, labels, dictionary = cv2.kmeans(desc, dictionary_size, None, criteria, 1, flags)
print("Time elapsed: {}s".format(get_elapsed_time(start)))

print("Getting histograms...")
start = cv2.getTickCount()
size = labels.shape[0] * labels.shape[1]
for i in xrange(size):
    label = labels[i]
    img_id = desc_src_img[i]
    data = imgs_data[img_id]
    data.bow_features[label] += 1
print("Time elapsed: {}s".format(get_elapsed_time(start)))

for data, img_file in zip(imgs_data, imgs):
    if img_file.name == test_img:
        print data.bow_features.ravel()

matcher.add(dictionary)
matcher.train()

for img_file in imgs:
    if img_file.name == test_img:
        descriptors = get_descriptors(img_file.img, detector)
        result = get_bow_features(matcher, descriptors, dictionary_size)
        print result.ravel()
