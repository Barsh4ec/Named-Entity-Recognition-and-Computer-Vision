import numpy as np
import cv2
import matplotlib.pyplot as plt


def select_descriptor_method(image):
    descriptor = cv2.SIFT_create(10000)
    kp = descriptor.detect(image, None)

    return descriptor.compute(image, kp)


def key_points_matching_KNN(features_train_img, features_query_img, ratio):

    bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=False)
    raw_matches = bf.knnMatch(features_train_img, features_query_img, k=2)
    print("Raw matches (knn):", len(raw_matches))
    matches = []

    for m, n in raw_matches:
        if m.distance < n.distance * ratio:
            matches.append(m)

    return matches


def match_features(train_photo, query_photo):
    train_photo = cv2.cvtColor(train_photo, cv2.COLOR_BGR2RGB)
    query_photo = cv2.cvtColor(query_photo, cv2.COLOR_BGR2RGB)

    train_photo_gray = cv2.cvtColor(train_photo, cv2.COLOR_RGB2GRAY)
    query_photo_gray = cv2.cvtColor(query_photo, cv2.COLOR_RGB2GRAY)

    train_photo = cv2.resize(train_photo, (2000, 2000))
    query_photo = cv2.resize(query_photo, (2000, 2000))
    train_photo_gray = cv2.resize(train_photo_gray, (2000, 2000))
    query_photo_gray = cv2.resize(query_photo_gray, (2000, 2000))

    keypoints_train_img, feature_train_img = select_descriptor_method(train_photo_gray)
    keypoints_query_img, feature_query_img = select_descriptor_method(query_photo_gray)

    fig = plt.figure(figsize=(20, 8))

    matches = key_points_matching_KNN(feature_train_img, feature_query_img, ratio=0.55)

    mapped_features_image_knn = cv2.drawMatches(
        train_photo,
        keypoints_train_img,
        query_photo,
        keypoints_query_img,
        np.random.choice(matches, 100),
        None,
        flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)

    plt.imshow(mapped_features_image_knn)
