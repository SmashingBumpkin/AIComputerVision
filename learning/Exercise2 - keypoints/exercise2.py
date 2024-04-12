# Prof solution to keypoint ex

import numpy as np
import cv2

images = []

path = "learning/Exercise2 - keypoints/"

img = cv2.imread(path + "comfort mob.jpg")
images.append(img)

img = cv2.imread(path + "green tin.webp")
images.append(img)

img = cv2.imread(path + "zero no sleep.jpg")
images.append(img)

classes = ["comfort mob", "green roast tin", "no sleep til shengal"]

orb = cv2.ORB_create()


def descriptorsDB(images):
    descriptors = []
    key_points = []
    for image in images:
        kpt, descriptor = orb.detectAndCompute(image, None)
        descriptors.append(descriptor)
        key_points.append(kpt)
    return descriptors, key_points


descriptors, key_points = descriptorsDB(images)

bf_matcher = cv2.BFMatcher(cv2.NORM_HAMMING)


def objClassification(img, descriptors):
    kpt, descriptor = orb.detectAndCompute(img, None)
    classification = -1
    classification_quality = -1
    for index, class_descriptor in enumerate(descriptors):
        matches = bf_matcher.knnMatch(descriptor, class_descriptor, k=2)
        good_matches = []
        for m, n in matches:  # K=2 so just two points needed for checkin
            if m.distance < 0.9 * n.distance:
                # Lowering the value improves the quality of matches
                good_matches.append(m)

        if len(good_matches) > classification_quality:
            classification_quality = len(good_matches)
            classification = index
    return classes[classification]


for i in range(3):

    test_image = cv2.imread(path + f"book{str(i+1)}.jpeg")

    print(objClassification(test_image, descriptors))
