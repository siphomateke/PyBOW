################################################################################

# functionality: parameter settings for detection algorithm training/testing

# This version: (c) 2018 Toby Breckon, Dept. Computer Science, Durham University, UK
# License: MIT License

# Origin acknowledgements: forked from https://github.com/nextgensparx/PyBOW

################################################################################

import cv2

################################################################################
# settings for datsets in general

# data location - training examples

DATA_training_path_neg = "pedestrain/INRIAPerson/train_64x128_H96/neg/"
DATA_training_path_pos = "pedestrain/INRIAPerson/train_64x128_H96/pos/"

# data location - testing examples

# DATA_testing_path_neg = "pedestrain/INRIAPerson/test_64x128_H96/neg/"
# DATA_testing_path_pos = "pedestrain/INRIAPerson/test_64x128_H96/pos/"

DATA_testing_path_neg = "pedestrain/INRIAPerson/test_64x128_H96/neg/"
DATA_testing_path_pos = "pedestrain/INRIAPerson/test_64x128_H96/pos/"

DATA_WINDOW_SIZE = [64, 128];

# class names - N.B. ordering of 0, 1 for neg/pos = order of paths

DATA_CLASS_NAMES = {
    "other": 0,
    "pedestrain": 1
}

################################################################################
# settings for BOW - Bag of (visual) Word - approaches

BOW_SVM_PATH = "svm_bow.xml"
BOW_DICT_PATH = "bow_dictionary.npy"

BOW_dictionary_size = 512;  # in general, larger = better performance, but potentially slower
BOW_SVM_kernel = cv2.ml.SVM_RBF; # see opencv manual for other options
BOW_SVM_max_training_iterations = 1000; # stop training after max iterations

# specify the type of featrue points to use
# -- refer to the OpenCV manual for options here, by default this is set to work on
# --- all systems "out of the box" rather than using the best available option

try:

    DETECTOR = cv2.xfeatures2d.SIFT_create() # -- requires extra modules and non-free build flag
    # DETECTOR = cv2.xfeatures2d.SURF_create() # -- requires extra modules and non-free build flag

    print("Features in use: ", DETECTOR.__class__())

    _algorithm = 0 # FLANN_INDEX_KDTREE
    _index_params = dict(algorithm=_algorithm, trees=5)
    _search_params = dict(checks=50)

except:
    # DETECTOR = cv2.AKAZE_create()
    # DETECTOR = cv2.KAZE_create()
    DETECTOR = cv2.ORB_create(nfeatures=100000, scoreType=cv2.ORB_FAST_SCORE) # check these params

    #if using ORB points
    # taken from: https://docs.opencv.org/3.3.0/dc/dc3/tutorial_py_matcher.html
    # N.B. "commented values are recommended as per the docs,
    # but it didn't provide required results in some cases"

    _algorithm = 6 # FLANN_INDEX_LSH
    _index_params= dict(algorithm = _algorithm,
                        table_number = 6, # 12
                        key_size = 12,     # 20
                        multi_probe_level = 1) #2
    _search_params = dict(checks=50)


    # SAPIENT ORB uses:
    # extractor = ORB::create(500, 1.2f, 8, 5, 0, 2, ORB::HARRIS_SCORE, 5, 20);

    print("Falling back to using features: ", DETECTOR.__class__())

# based on choice and availability of feature points, set up KD-tree matcher

MATCHER = cv2.FlannBasedMatcher(_index_params, _search_params)

################################################################################
# settings for HOG approaches

HOG_BIN_N = 16

HOG_SVM_PATH = "svm_hog.xml"

################################################################################
