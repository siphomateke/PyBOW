import os

import cv2


def imlist(path):
    """
    The function imlist returns all the names of the files in
    the directory path supplied as argument to the function.
    """
    return [os.path.join(path, f) for f in os.listdir(path)]


class ImageFile(object):
    def __init__(self, name, img):
        self.name = name
        self.img = img


def imreads(path):
    """
    This reads all the images in a given folder and returns the results
    """
    images_path = [os.path.join(path, f) for f in os.listdir(path)]
    images = []
    for image_path in images_path:
        img = cv2.imread(image_path)
        name = os.path.basename(image_path)
        images.append(ImageFile(name, img))
    return images
