################################################################################

# functionality: parameter settings for detection algorithms

# This version: (c) 2018 Toby Breckon, Dept. Computer Science, Durham University, UK
# License: MIT License (https://github.com/tobybreckon/python-bow-hog-object-detection/blob/master/LICENSE)

# Origin ackowledgements: forked from https://github.com/nextgensparx/PyBOW
# but code portions may have broader origins elsewhere also it appears

################################################################################

import cv2

################################################################################

# settings for BOW approaches

BOW_MAX_IMG_WIDTH = 320
BOW_SVM_PATH = "svm_bow.xml"
BOW_DICT_PATH = "bow_dictionary.npy"
BOW_CLASS_NAMES = {
    "other": 0,
    "pedestrain": 1
}

# algorithm = FLANN_INDEX_KDTREE

_index_params = dict(algorithm=0, trees=5)
_search_params = dict(checks=50)

MATCHER = cv2.FlannBasedMatcher(_index_params, _search_params)
DETECTOR = cv2.AKAZE_create()
#DETECTOR = cv2.KAZE_create()
#DETECTOR = cv2.ORB_create(nfeatures=100000, scoreType=cv2.ORB_FAST_SCORE)

################################################################################

# settings for HOG approaches

HOG_BIN_N = 16

HOG_SVM_PATH = "svm_hog.xml"
HOG_CLASS_NAMES = {
    "other": 0,
    "pedestrain": 1
}

################################################################################
