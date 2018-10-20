################################################################################

# functionality: perform detection based on Bag of (visual) Words SVM classification
# using a very basic multi-scale, sliding window (exhaustive search) approach

# This version: (c) 2018 Toby Breckon, Dept. Computer Science, Durham University, UK
# License: MIT License (https://github.com/tobybreckon/python-bow-hog-object-detection/blob/master/LICENSE)

# Origin ackowledgements: forked from https://github.com/nextgensparx/PyBOW

################################################################################

import numpy as np
import cv2
import os
import math
from utils import resize_img
from utils import ImageData
import params

################################################################################

directory_to_cycle = params.DATA_testing_path_pos;

show_scan_window_process = True;

################################################################################

# a very basic approach to produce an image at multi-scales (i.e. variant
# re-sized resolutions) - can be easily improved upon using OpenCV - **TODO** REF

def pyramid(img, scale=1.5, min_size=(30, 30)):
    # yield the original image
    yield img

    # keep looping over the pyramid
    while True:
        # compute the new dimensions of the image and resize it
        w = int(img.shape[1] / scale)
        img = resize_img(img, width=w)

        # if the resized image does not meet the supplied minimum
        # size, then stop constructing the pyramid
        if img.shape[0] < min_size[1] or img.shape[1] < min_size[0]:
            break

        # yield the next image in the pyramid
        yield img

################################################################################

# generate a set of sliding window locations across the image

def sliding_window(image, window_size, step_size=8):
    # slide a window across the image
    for y in range(0, image.shape[0], step_size):
        for x in range(0, image.shape[1], step_size):
            # yield the current window
            window = image[y:y + window_size[1], x:x + window_size[0]]
            if not (window.shape[0] != window_size[1] or window.shape[1] != window_size[0]):
                yield (x, y, window)

################################################################################

# perform basic non-maximal suppression of overlapping object detections

def non_max_suppression_fast(boxes, overlapThresh):
    # if there are no boxes, return an empty list
    if len(boxes) == 0:
        return []

    # if the bounding boxes integers, convert them to floats --
    # this is important since we'll be doing a bunch of divisions
    if boxes.dtype.kind == "i":
        boxes = boxes.astype("float")

    # initialize the list of picked indexes
    pick = []

    # grab the coordinates of the bounding boxes
    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]

    # compute the area of the bounding boxes and sort the bounding
    # boxes by the bottom-right y-coordinate of the bounding box
    area = (x2 - x1 + 1) * (y2 - y1 + 1)
    idxs = np.argsort(y2)

    # keep looping while some indexes still remain in the indexes
    # list
    while len(idxs) > 0:
        # grab the last index in the indexes list and add the
        # index value to the list of picked indexes
        last = len(idxs) - 1
        i = idxs[last]
        pick.append(i)

        # find the largest (x, y) coordinates for the start of
        # the bounding box and the smallest (x, y) coordinates
        # for the end of the bounding box
        xx1 = np.maximum(x1[i], x1[idxs[:last]])
        yy1 = np.maximum(y1[i], y1[idxs[:last]])
        xx2 = np.minimum(x2[i], x2[idxs[:last]])
        yy2 = np.minimum(y2[i], y2[idxs[:last]])

        # compute the width and height of the bounding box
        w = np.maximum(0, xx2 - xx1 + 1)
        h = np.maximum(0, yy2 - yy1 + 1)

        # compute the ratio of overlap
        overlap = (w * h) / area[idxs[:last]]

        # delete all indexes from the index list that have
        idxs = np.delete(idxs, np.concatenate(([last],
                                               np.where(overlap > overlapThresh)[0])))

    # return only the bounding boxes that were picked using the
    # integer data type
    return boxes[pick].astype("int")

####################################################################################

# load dictionary and SVM data

dictionary = np.load(params.BOW_DICT_PATH)
svm = cv2.ml.SVM_load(params.BOW_SVM_PATH)

####################################################################################

# process all images in directory (sorted by filename)

for filename in sorted(os.listdir(directory_to_cycle)):

    # if it is a PNG file

    if '.png' in filename:
        print(os.path.join(directory_to_cycle, filename));

        # read image data

        image = cv2.imread(os.path.join(directory_to_cycle, filename), cv2.IMREAD_COLOR);

        # make a copy for drawing the output

        output = image.clone();

        # for a range of different image scales in an image pyramid

        current_scale = -1
        for resized in pyramid(image, scale=1.25):
            if current_scale == -1:
                current_scale = 1
            else:
                current_scale /= 1.25
            rect_img = resized.copy()

            if (show_scan_window_process):
                cv2.imshow('current scale',rect_img)

            # loop over the sliding window for each layer of the pyramid (re-sized image)

            step = math.floor(resized.shape[0] / 16)
            window_size = params.DATA_WINDOW_SIZE;
            detections = []
            if step > 0:
                for (x, y, window) in sliding_window(resized, window_size, step_size=step):

                    if (show_scan_window_process):
                        cv2.imshow('current window',window)
                        key = cv2.waitKey(10) # wait 10ms

                    # for each window region get the BoW feature point descriptors

                    img_data = ImageData(window)
                    img_data.compute_descriptors()

                    # generate and classify each window by constructing a BoW
                    # histogram and passing it through the SVM classifier

                    if img_data.descriptors is not None:
                        img_data.generate_bow_hist(dictionary)

                        results = svm.predict(np.float32([img_data.features]))
                        output = results[1].ravel()[0]

                        print(output," ", params.DATA_CLASS_NAMES["pedestrain"]);

                        # if we get a detection, then record it

                        if output == params.DATA_CLASS_NAMES["pedestrain"]:
                            rect = np.float32([x, y, x + window_size[0], y + window_size[1]])
                            rect *= (1.0 / current_scale) # ???
                            detections.append(rect)

                # for the overall set of detections perform non maximal suppression
                # (i.e. remove overlapping boxes etc)
                # for multi-class problems this could be improved by also using
                # the distance from the SVM decision boundary (float return value of SVM)

                detections = non_max_suppression_fast(np.int32(detections), 0.4)
                detections = np.int32(detections)

                # draw all the detection on the original image

                for rect in detections:
                    cv2.rectangle(output, (rect[0], rect[1]), (rect[0] + rect[2], rect[1] + rect[3]), (0, 0, 255), 2)

            # display the image

            cv2.imshow('original image',output)
            key = cv2.waitKey(200) # wait 200ms - N.B. limited this loop to < 5 fps
            if (key == ord('x')):
                break

####################################################################################

# close all windows

cv2.destroyAllWindows()

####################################################################################
