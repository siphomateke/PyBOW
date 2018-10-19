################################################################################

# functionality: ....

# This version: (c) 2018 Toby Breckon, Dept. Computer Science, Durham University, UK
# License: MIT License (https://github.com/tobybreckon/python-bow-hog-object-detection/blob/master/LICENSE)

# Origin ackowledgements: forked from https://github.com/nextgensparx/PyBOW
# but code portions may have broader origins elsewhere also it appears

################################################################################

import cv2
from utils import *


def generate_dictionary(imgs_data, dictionary_size):
    # Extracting descriptors
    desc = stack_array([img_data.descriptors for img_data in imgs_data])
    # important, cv2.kmeans only accepts type32 descriptors
    desc = np.float32(desc)

    # Clustering
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 0.01)
    flags = cv2.KMEANS_PP_CENTERS
    # desc is a type32 numpy array of vstacked descriptors
    compactness, labels, dictionary = cv2.kmeans(desc, dictionary_size, None, criteria, 1, flags)
    np.save(params.DICT_PATH, dictionary)

    return dictionary


def main():
    dictionary_size = 512
    # Loading images
    """imgs_data = []  # type: list[ImageData]

    pos_imgs_path = "train/pos"
    neg_imgs_path = "train/neg"

    print("Loading images...")

    # imreads returns a list of all images in that directory
    pos_imgs = imreads(pos_imgs_path)
    neg_imgs = imreads(neg_imgs_path)

    img_count = 0
    for img in pos_imgs:
        img_data = ImageData(img)
        img_data.set_class("pos")
        imgs_data.insert(img_count, img_data)
        img_count += 1

    for img in neg_imgs:
        img_data = ImageData(img)
        img_data.set_class("neg")
        imgs_data.insert(img_count, img_data)
        img_count += 1"""

    program_start = cv2.getTickCount()

    print("Loading images...")
    start = cv2.getTickCount()
    paths = ["train/pos", "train/neg"]
    class_names = ["pos", "neg"]
    imgs_data = get_imgs_data(paths, class_names)
    print("Loaded {} image(s)".format(len(imgs_data)))
    print_duration(start)

    print("Computing descriptors...")
    start = cv2.getTickCount()
    [img_data.compute_descriptors() for img_data in imgs_data]
    print_duration(start)

    print("Clustering...")
    start = cv2.getTickCount()
    dictionary = generate_dictionary(imgs_data, dictionary_size)
    print_duration(start)

    print("Generating histograms...")
    start = cv2.getTickCount()
    [img_data.generate_bow_hist(dictionary) for img_data in imgs_data]
    print_duration(start)

    print imgs_data[0].hog().shape
    print imgs_data[0].features.shape

    print("Training SVM...")
    start = cv2.getTickCount()
    # Begin training SVM
    svm = cv2.ml.SVM_create()
    svm.setType(cv2.ml.SVM_C_SVC)
    svm.setKernel(cv2.ml.SVM_LINEAR)
    svm.setC(2.67)
    svm.setGamma(5.383)

    # Compile samples
    samples = get_samples(imgs_data)
    responses = np.int32([img_data.response for img_data in imgs_data])

    svm.setTermCriteria((cv2.TERM_CRITERIA_COUNT, 1000, 1.e-06))
    svm.train(samples, cv2.ml.ROW_SAMPLE, responses)
    svm.save(params.SVM_PATH)

    output = svm.predict(samples)[1].ravel()
    error = (np.absolute(responses.ravel() - output).sum()) / float(output.shape[0])

    if error < 0.2:
        print "Successfully trained SVM with {}% error".format(error * 100)
    else:
        print "Failed to train SVM. {}% error".format(error * 100)
    print_duration(start)

    print("Finished training BOW detector. {}".format(format_time(get_elapsed_time(program_start))))


if __name__ == '__main__':
    main()
