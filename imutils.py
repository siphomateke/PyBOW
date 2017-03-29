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

    def to_gray(self):
        self.img = cv2.cvtColor(self.img, cv2.COLOR_BGR2GRAY)


def resize_img_files(img_files, width=-1, height=-1):
    """
    Resizes a list of ImageFiles to a specified width or height while keeping the aspect ratio
    """
    if height == -1 and width == -1:
        raise TypeError("Invalid arguments. Width or height must be provided.")
    for i in range(len(img_files)):
        img = img_files[i].img
        h = img.shape[0]
        w = img.shape[1]
        if height == -1:
            aspect_ratio = float(w) / h
            new_height = int(width / aspect_ratio)
            img_files[i].img = cv2.resize(img, (width, new_height))
        elif width == -1:
            aspect_ratio = h / float(w)
            new_width = int(height / aspect_ratio)
            img_files[i].img = cv2.resize(img, (new_width, height))


def imreads(path):
    """
    This reads all the images in a given folder and returns the results
    """
    images_path = imlist(path)
    images = []
    for image_path in images_path:
        img = cv2.imread(image_path)
        name = os.path.basename(image_path)
        images.append(ImageFile(name, img))
    return images
