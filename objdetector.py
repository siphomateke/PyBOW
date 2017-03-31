import numpy as np
import cv2
from bowutils import resize_img
from bowutils import get_imgs_data
from bowutils import get_samples
from bowutils import get_responses
from bowutils import ImageData
import params
from matplotlib import pyplot as plt


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


def sliding_window(image, window_size, step_size=8):
    # slide a window across the image
    for y in xrange(0, image.shape[0], step_size):
        for x in xrange(0, image.shape[1], step_size):
            # yield the current window
            window = image[y:y + window_size[1], x:x + window_size[0]]
            if not (window.shape[0] != window_size[1] or window.shape[1] != window_size[0]):
                yield (x, y, window)


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


image = cv2.imread("test/pos/test8.jpg")
window_size = (320, 240)

dictionary = np.load(params.DICT_PATH)
svm = cv2.ml.SVM_load(params.SVM_PATH)

detections = []
current_scale = -1
for resized in pyramid(image, scale=1.25):
    if current_scale == -1:
        current_scale = 1
    else:
        current_scale /= 1.25
    rect_img = resized.copy()
    # loop over the sliding window for each layer of the pyramid
    step = (resized.shape[0] / window_size[0]) * 16
    if step > 0:
        for (x, y, window) in sliding_window(resized, window_size, step_size=step):

            img_data = ImageData(window)
            img_data.compute_descriptors()

            if img_data.descriptors is not None:
                img_data.generate_bow_hist(dictionary)

                results = svm.predict(np.float32([img_data.features]))
                output = results[1].ravel()[0]

                if output == 0.0:
                    rect = np.float32([x, y, x + window_size[0], y + window_size[1]])
                    rect *= (1.0 / current_scale)
                    detections.append(rect)
                    cv2.rectangle(rect_img, (x, y), (x + window_size[0], y + window_size[1]), (0, 0, 255), 2)

            clone = rect_img.copy()
            cv2.rectangle(clone, (x, y), (x + window_size[0], y + window_size[1]), (0, 255, 0), 2)
            if clone.shape[0] > params.MAX_IMG_WIDTH:
                clone = resize_img(clone, params.MAX_IMG_WIDTH)
            cv2.imshow("Window", clone)
            cv2.waitKey(1)

detections = non_max_suppression_fast(np.int32(detections), 0.4)
detections = np.int32(detections)
rect_img = image.copy()
for rect in detections:
    cv2.rectangle(rect_img, (rect[0], rect[1]), (rect[0] + rect[2], rect[1] + rect[3]), (0, 0, 255), 2)

if len(detections)>0:
    cv2.imshow("Window", resize_img(rect_img, 640))
    cv2.waitKey(0)
