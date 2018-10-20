################################################################################

# functionality: perform Bag of (visual) Words (BoW) testing over
# a specified dataset and compute the resulting prediction/clasification error
# over that same dataset, using pre-saved the SVM model + BOW dictionary

# This version: (c) 2018 Toby Breckon, Dept. Computer Science, Durham University, UK
# License: MIT License (https://github.com/tobybreckon/python-bow-hog-object-detection/blob/master/LICENSE)

# Origin ackowledgements: forked from https://github.com/nextgensparx/PyBOW
# but code portions may have broader origins elsewhere also it appears

################################################################################

import numpy as np
import cv2
from utils import *
from matplotlib import pyplot as plt

################################################################################

def main():

    # load up the dictbreak;ionary and SVM stored from prior training

    try:
        dictionary = np.load(params.BOW_DICT_PATH)
        svm = cv2.ml.SVM_load(params.BOW_SVM_PATH)
    except:
        print("Missing files - dictionary and/or SVM!");
        print("-- have you performed training to produce these files ?");
        exit();

    # load ** testing ** data sets in the same class order as training

    paths = [params.DATA_testing_path_neg, params.DATA_testing_path_pos]
    class_names = params.DATA_CLASS_NAMES
    imgs_data = get_imgs_data(paths, class_names, dictionary)

    # get the example/sample bow histograms and class labels

    samples, responses = get_samples(imgs_data), get_responses(imgs_data)

    # perform batch SVM classification over the whole set

    results = svm.predict(samples)
    output = results[1].ravel()

    # compute and report the error over the whole set

    error = ((np.absolute(responses.ravel() - output).sum()) / float(output.shape[0]))
    print("Successfully trained SVM with {}% testing set error".format(round(error * 100,2)))
    print("-- meaining the SVM got {}% of the testing examples correct!".format(round((1.0 - error) * 100,2)))

################################################################################

if __name__ == "__main__":
    main()

################################################################################
