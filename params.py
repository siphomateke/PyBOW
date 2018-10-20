################################################################################

# functionality: parameter settings for detection algorithm training/testing

# This version: (c) 2018 Toby Breckon, Dept. Computer Science, Durham University, UK
# License: MIT License (https://github.com/tobybreckon/python-bow-hog-object-detection/blob/master/LICENSE)

# Origin ackowledgements: forked from https://github.com/nextgensparx/PyBOW

################################################################################

import cv2

################################################################################
# settings for datsets in general

# data location - training examples

DATA_training_path_neg = "pedestrain/INRIAPerson/Train/neg/"
DATA_training_path_pos = "pedestrain/INRIAPerson/Train/pos/"

# data location - testing examples

DATA_testing_path_neg = "pedestrain/INRIAPerson/Test/neg/"
DATA_testing_path_pos = "pedestrain/INRIAPerson/Test/pos/"

# class names - N.B. ordering of 0, 1 for neg/pos = order of paths

DATA_CLASS_NAMES = {
    "other": 0,
    "pedestrain": 1
}

################################################################################
# settings for BOW - Bag of (visual) Word - approaches

BOW_SVM_PATH = "svm_bow.xml"
BOW_DICT_PATH = "bow_dictionary.npy"

BOW_dictionary_size = 512   # in general, larger = better performance, but potentially slower

# settings for algorithm = FLANN_INDEX_KDTREE

_index_params = dict(algorithm=0, trees=5)
_search_params = dict(checks=50)
MATCHER = cv2.FlannBasedMatcher(_index_params, _search_params)

# specify the type of featrue points to use
# -- refer to the OpenCV manual for options here, by default this is set to work on
# --- all systems "out of the box" rather than using the best available option

DETECTOR = cv2.AKAZE_create()
#DETECTOR = cv2.KAZE_create()
#DETECTOR = cv2.ORB_create(nfeatures=100000, scoreType=cv2.ORB_FAST_SCORE) # check these params

# DETECTOR = cv2.xfeatures2d.SIFT_create() # -- requires extra modules and non-free build flag
# DETECTOR = cv2.xfeatures2d.SURF_create() # -- requires extra modules and non-free build flag

################################################################################
# settings for HOG approaches

HOG_BIN_N = 16

HOG_SVM_PATH = "svm_hog.xml"

################################################################################
