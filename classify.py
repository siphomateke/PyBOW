import numpy as np
import cv2
from bowutils import *
from matplotlib import pyplot as plt

def main():
    # Load up the dictionary
    dictionary = np.load(params.DICT_PATH)

    paths = ["test/pos", "test/neg"]
    class_names = ["pos", "neg"]
    imgs_data = get_imgs_data(paths, class_names, dictionary)

    samples, responses = get_samples(imgs_data), get_responses(imgs_data)

    svm = cv2.ml.SVM_load(params.SVM_PATH)
    results = svm.predict(samples)
    output = results[1].ravel()

    error = ((np.absolute(responses.ravel() - output).sum()) / float(output.shape[0])) * 100
    print "Error in test data: {}%".format(error)

    for i in xrange(len(imgs_data)):
        img_data = imgs_data[i]

        title = "Prediction: {0}".format(output[i])
        plt.axis("off")
        plt.imshow(cv2.cvtColor(img_data.img, cv2.COLOR_BGR2RGB))
        plt.suptitle(title)
        plt.draw()
        plt.waitforbuttonpress(0)  # this will wait for indefinite time
        plt.clf()


if __name__ == "__main__":
    main()
