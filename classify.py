# import numpy as np
import cv2
from bowutils import *


def add_to_imgs_data(path, class_name, imgs_data):
    imgs = imreads(path)

    img_count = len(imgs_data)
    for img in imgs:
        img_data = ImageData(img)
        img_data.set_class(class_name)
        imgs_data.insert(img_count, img_data)
        img_count += 1

    return imgs_data


def get_imgs_data(paths, class_names, dictionary):
    imgs_data = []  # type: list[ImageData]

    for path, class_name in zip(paths, class_names):
        add_to_imgs_data(path, class_name, imgs_data)

    [img_data.compute_descriptors(params.DETECTOR) for img_data in imgs_data]
    [img_data.generate_bow_hist(params.MATCHER, dictionary) for img_data in imgs_data]

    return imgs_data


def main():
    # Load up the dictionary
    dictionary = np.load("ml/dictionary.npy")

    paths = ["test/pos"]
    class_names = ["pos"]
    imgs_data = get_imgs_data(paths, class_names, dictionary)

    samples, responses = get_samples(imgs_data), get_responses(imgs_data)

    svm = cv2.ml.SVM_create()
    svm.load("ml/svm.xml")
    output = svm.predict(samples)[1].ravel()

    error = ((np.absolute(responses.ravel() - output).sum()) / float(output.shape[0])) * 100
    print "Error in test data: {}%".format(error)


if __name__ == "__main__":
    main()
