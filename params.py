import cv2

SVM_PATH = "ml/svm.xml"
CLASS_NAMES = {
    "pos": 0,
    "neg": 1
}

# algorithm = FLANN_INDEX_KDTREE
_index_params = dict(algorithm=0, trees=5)
_search_params = dict(checks=50)

MATCHER = cv2.FlannBasedMatcher(_index_params, _search_params)
DETECTOR = cv2.AKAZE_create()
