import cv2
import numpy as np
import imutils


def get_elapsed_time(start):
    """ Helper function for timing code execution"""
    return (cv2.getTickCount() - start) / cv2.getTickFrequency()


def resize_img_files(img_files, width=-1, height=-1):
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


def to_gray(color_img):
    gray = cv2.cvtColor(color_img, cv2.COLOR_BGR2GRAY)
    return gray


FLANN_INDEX_KDTREE = 1  # bug: flann enums are missing
FLANN_INDEX_LSH = 6


def get_detector_matcher(name, use_flann):
    if name == 'sift':
        detector = cv2.xfeatures2d.SIFT_create()
        norm = cv2.NORM_L2
    elif name == 'surf':
        detector = cv2.xfeatures2d.SURF_create(800)
        norm = cv2.NORM_L2
    elif name == 'orb':
        detector = cv2.ORB_create(400)
        norm = cv2.NORM_HAMMING
    elif name == 'akaze':
        detector = cv2.AKAZE_create()
        norm = cv2.NORM_HAMMING
    elif name == 'brisk':
        detector = cv2.BRISK_create()
        norm = cv2.NORM_HAMMING
    else:
        return None, None
    if use_flann:
        if norm == cv2.NORM_L2:
            flann_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
        else:
            flann_params = dict(algorithm=FLANN_INDEX_LSH,
                                table_number=6,  # 12
                                key_size=12,  # 20
                                multi_probe_level=1)  # 2
        matcher = cv2.FlannBasedMatcher(flann_params, {})  # bug : need to pass empty dict (#1329)
    else:
        matcher = cv2.BFMatcher(norm)
    return detector, matcher


def get_descriptors(img, detector):
    descriptors = detector.detectAndCompute(img, None)[1]
    return descriptors


def get_batch_descriptors(img_files, detector):
    all_descriptors = np.array([])
    descriptor_img_ids = []
    i = 0
    for imgFile in img_files:
        img = imgFile.img
        descriptors = get_descriptors(img, detector)
        # vewy important, cv2.kmeans only accepts type32 descriptors
        if len(all_descriptors) == 0:
            all_descriptors = np.array(descriptors)
        else:
            all_descriptors = np.vstack((all_descriptors, descriptors))
        for j in range(len(descriptors)):
            descriptor_img_ids.append(i)
        i += 1
    all_descriptors = np.float32(all_descriptors)
    return all_descriptors, descriptor_img_ids


class ImageData(object):
    def __init__(self, size, class_name):
        self.bow_features = np.zeros((size, 1))
        self.class_name = class_name


def get_bow_features(matcher, descriptors, dictionary_size):
    output = np.zeros((dictionary_size, 1), np.float32)
    matches = matcher.match(np.float32(descriptors))
    for match in matches:
        visual_word = match.trainIdx
        output[visual_word] += 1
    return output


def get_img_samples(img_files, detector, matcher, dictionary_size):
    """

    :type imgs_data: list[ImageData]
    :type dictionary_size: int
    """

    resize_img_files(img_files, 320)
    [imgFile.to_gray() for imgFile in img_files]

    samples = np.array([])
    for i in xrange(len(img_files)):
        img = img_files[i].img
        descriptors = get_descriptors(img, detector)
        bow_features = get_bow_features(matcher, descriptors, dictionary_size)
        bow_features = cv2.normalize(bow_features, None, 0, bow_features.shape[0], cv2.NORM_MINMAX, -1)

        if len(samples) == 0:
            samples = np.array([bow_features])
        else:
            samples = np.vstack((samples, [bow_features]))
    return samples


dictionary_size = 512
pos_imgs_path = "train\pos"
neg_imgs_path = "train\/neg"
test_img = "[name of image to compare descriptors for]"

print("Loading images...")
all_imgs_data = []  # type: list[ImageData]
all_imgs = []
pos_imgs = imutils.imreads(pos_imgs_path) # type: list[imutils.ImageFile]
for imgFile in pos_imgs:
    all_imgs_data.append(ImageData(dictionary_size, 0))
    all_imgs.append(imgFile)
neg_imgs = imutils.imreads(neg_imgs_path)
for imgFile in neg_imgs:
    all_imgs_data.append(ImageData(dictionary_size, 1))
    all_imgs.append(imgFile)

# Resizes images to a fixed width
resize_img_files(all_imgs, 320)
# Converts images to their grayscale equivalent
[imgFile.to_gray() for imgFile in all_imgs]

print("Extracting features...")
start = cv2.getTickCount()
detector, matcher = get_detector_matcher("akaze", False)
# descriptor_imgs is a list which says what the id of each descriptors image is in all_imgs
desc, desc_src_img = get_batch_descriptors(all_imgs, detector)
print("Time elapsed: {}s".format(get_elapsed_time(start)))

# desc = np.vstack((pos_descriptors, neg_descriptors))
# desc = np.reshape(desc, (len(desc) / 61, 61))
# desc = np.float32(descriptors)

print("Clustering...")
start = cv2.getTickCount()
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 0.01)
flags = cv2.KMEANS_PP_CENTERS
compactness, labels, dictionary = cv2.kmeans(desc, dictionary_size, None, criteria, 1, flags)
print("Time elapsed: {}s".format(get_elapsed_time(start)))

print("Getting histograms...")
start = cv2.getTickCount()
size = labels.shape[0] * labels.shape[1]
for i in xrange(size):
    label = labels[i]
    img_id = desc_src_img[i]
    data = all_imgs_data[img_id]
    data.bow_features[label] += 1
print("Time elapsed: {}s".format(get_elapsed_time(start)))

print("Training SVM...")
start = cv2.getTickCount()
svm = cv2.ml.SVM_create()
svm.setType(cv2.ml.SVM_C_SVC)
svm.setKernel(cv2.ml.SVM_LINEAR)

svm.setTermCriteria((cv2.TERM_CRITERIA_COUNT, 100, 1.e-06))

samples = np.array([])
responses = np.array([])
count = 0
for data in all_imgs_data:
    if all_imgs[count].name == "Achaea lienardi, Noctuidae, Catocalinae M-2006-084.jpg":
        print data.bow_features.ravel()
    bow_feature = cv2.normalize(data.bow_features, None, 0, data.bow_features.shape[0], cv2.NORM_MINMAX, -1)
    if len(samples) == 0:
        samples = np.array([bow_feature])
    else:
        samples = np.vstack((samples, [bow_feature]))

    if len(responses) == 0:
        responses = np.array([data.class_name])
    else:
        responses = np.vstack((responses, data.class_name))
    count += 1
samples = np.float32(samples)
svm.train(samples, cv2.ml.ROW_SAMPLE, responses)
print("Time elapsed: {}s".format(get_elapsed_time(start)))

print("Training Matcher...")
start = cv2.getTickCount()
# 2 0 0 7 15 7 1
"""flann = cv2.FlannBasedMatcher_create()
flann.add(dictionary)
flann.train()"""
matcher.add(dictionary)
matcher.train()
print("Time elapsed: {}s".format(get_elapsed_time(start)))

for imgFile in all_imgs:
    if imgFile.name == "Achaea lienardi, Noctuidae, Catocalinae M-2006-084.jpg":
        descriptors = get_descriptors(imgFile.img, detector)
        result = get_bow_features(matcher, descriptors, dictionary_size)
        print result.ravel()

imgs2 = imutils.imreads("test\pos")
imgs2_data = []  # type: list[ImageData]
for imgFile in imgs2:
    imgs2_data.append(ImageData(dictionary_size, 0))

samples = get_img_samples(imgs2, detector, matcher, dictionary_size)
responses = np.array([])
for data in imgs2_data:
    if len(responses) == 0:
        responses = np.array([data.class_name])
    else:
        responses = np.vstack((responses, data.class_name))

output = svm.predict(samples)[1].ravel()
print responses.ravel()
print output
error = (np.absolute(responses.ravel() - output).sum() / float(output.shape[0])) * 100  # in percent
print "Error in training data: {}%".format(error)

"""for i in xrange(len(imgs2)):
    img = imgs2[i]
    plt.imshow(img)
    plt.title(output[i])
    plt.show()"""
