#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Created by jeffw on 2018/5/25


import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn import svm
from kaggle.digit_recognizer.global_config import GlobalConfig
from sklearn.externals import joblib


def display_an_image(label, image):
    img = image.values
    img = img.reshape((28, 28))
    plt.imshow(img, cmap='gray')
    plt.title('label=%d' % label)
    plt.show()


def display_an_dist(image):
    plt.hist(image)
    plt.show()


# dataset
labeled_images = pd.read_csv(GlobalConfig.INPUT_DIR + 'train.csv')
images = labeled_images.iloc[:, 1:]
labels = labeled_images.iloc[:, :1]
train_images, test_images, train_labels, test_labels = train_test_split(images, labels, train_size=0.8, random_state=0)
test_images[test_images > 0] = 1
train_images[train_images > 0] = 1

# # da
# display_an_image(train_labels.iloc[1, 0], train_images.iloc[1])
# display_an_dist(train_images.iloc[1])

# train & test
clf = svm.SVC()
clf.fit(train_images, train_labels.values.ravel())
print clf.score(test_images, test_labels)

# dump
joblib.dump(clf, GlobalConfig.MODEL_DIR + 'clf.model')


if __name__ == '__main__':
    pass
