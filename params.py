import cv2

MAX_IMG_WIDTH = 320
SVM_PATH = "ml/svm.xml"
DICT_PATH = "ml/dictionary.npy"
CLASS_NAMES = {
    "pos": 0,
    "neg": 1
}

# algorithm = FLANN_INDEX_KDTREE
_index_params = dict(algorithm=0, trees=5)
_search_params = dict(checks=50)

MATCHER = cv2.FlannBasedMatcher(_index_params, _search_params)
DETECTOR = cv2.AKAZE_create()
#DETECTOR = cv2.ORB_create(nfeatures=100000, scoreType=cv2.ORB_FAST_SCORE)
