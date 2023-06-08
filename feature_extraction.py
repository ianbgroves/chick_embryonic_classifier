# importing various libraries
import mahotas as mh
import numpy as np
from pylab import imshow, show
from binary_utils import *

import seaborn
import numpy as np
import matplotlib.pyplot as plt
import os
import cv2
import time
import random
import tensorflow
import pandas as pd
import sklearn

from pathlib import Path
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical

# Import functions from utils.py
from binary_utils import create_data, reshape_and_normalize, scree_plot, fit_PCA, elbow_plot, fit_k_means, counter, plotter, plot_counts, plot_scatter
# loading images
load = False
os.chdir(r'G:\My Drive\Python_projects\classifier\binary_classification\binary_brain_pca_kmeans_haralick')
if load:
  split_dict = load_test_set("G:/My Drive/Python_projects/classifier/binary_classification/saved_test_sets/binary_baseline_3_Mar-14-2023/pkl_splits")
  print(split_dict.keys())
  X = split_dict['X']
  Y = split_dict['Y']
  X_test = split_dict['X_test']
  y_test = split_dict['y_test']
  print(len(X))
  print(len(Y))
  print(len(X_test))
  print(len(y_test))
  print(
    "ratios of labels in the data set are {} {} {}".format(round(Y.count(0) / len(Y), 2),
                                                           round(Y.count(1) / len(Y), 2),
                                                           round(Y.count(2) / len(Y), 2)))
  print("ratios of labels in the test set are {} : {} : {}".format(round(y_test.count(0) / len(y_test), 2),
                                                                   round(y_test.count(1) / len(y_test), 2),
                                                                   round(y_test.count(2) / len(y_test), 2)))
else:

    data = create_data('data_10a_b', duplicate_channels=False, equalize=False)

    data_list = []
    data_list.append(data[0:len(data)])

    X = []
    Y = []
    for i in data_list:
      for feature, label in i:
        X.append(feature)
        Y.append(label)

X, Y = aug_data_2(X, Y)

def haralick_extraction(image, label):

    img_filtered = image
    # plt.imshow(img_filtered)
    # plt.show()

    #filter the image
    # img_filtered = img[:, :, 0]
    # plt.imshow(img_filtered)
    # plt.show()

    #gaussian filter
    img_filtered = mh.gaussian_filter(img_filtered, 4)

    # plt.imshow(img_filtered)
    # plt.show()

    #set threshold
    thresholded_img = (img_filtered > img_filtered.mean())
    # thresholded_img = img_filtered
    # plt.imshow(thresholded_img)
    # plt.show()

    #making a labelled image
    labeled, n = mh.label(thresholded_img)
    # plt.imshow(labeled)
    # plt.show()
    # print(n)

    lbp_features = mh.features.lbp(labeled, 100, 5)
    # plt.hist(lbp_features)
    # print('lbp features shape {}'.format(lbp_features.shape))
    # print('lbp features {}'.format(lbp_features))
    # plt.show()

    haralick_features = mh.features.haralick(labeled)
    # print('Haralick features shape {}'.format(haralick_features.shape))
    # print('Haralick features {}'.format(haralick_features))
    # plt.imshow(haralick_features)
    # plt.show()

    return haralick_features, lbp_features

haralick_features_list = []
lbp_features_list = []

haralick_labels = []
lbp_labels = []

for i in range(0,len(X)):
    img = X[i]
    label = Y[i]
    haralick_features, lbp_features = haralick_extraction(img, label)
    haralick_features_list.append(haralick_features)
    haralick_labels.append(label)

    lbp_features_list.append(lbp_features)
    lbp_labels.append(label)

print(len(X))
print(haralick_features_list[0])
print(haralick_labels[0])
print("haralick_shape_{}".format(haralick_features_list[0].shape))

#Sub directories for different categories
CATEGORIES = ["10a","10b"]

# print('Path:', path)
# print('Data directory:', DATADIR)

# X, Y = create_training_data_k_means()

# X_train, Y_train = reshape_and_normalize(X, Y, nb_classes=2)

def reshape_and_normalize_haralick(X, Y, nb_classes):

    X = np.array(X).reshape(-1, 4, 13, 1)  # Reshape Haralick features
    X_train = np.array(X)
    Y_train = np.array(Y)
    print("The shape of X is " + str(X_train.shape))
    print("The shape of y is " + str(Y_train.shape))  # This is only used to check our clustering

    # Data Normalization

    print("Pre normalisation X_train min is" + str(np.min(X_train)))  # Should be 0
    print(np.max(X_train))  # Should be some positive value

    # Conversion to float

    X_train = X_train.astype('float32')

    # Normalization

    X_train = (X_train - np.min(X_train)) / (np.max(X_train) - np.min(X_train))
    print(X_train.min())  # Should be 0
    print(X_train.max())  # Should be 1.0
    print(X_train[0])
    # check that X has been correctly split into train and test sets
    print('X_train shape:', X_train.shape)
    print(X_train.shape[0], 'train samples')

    # convert class vectors to binary class matrices with one-hot encoding

    Y_train = to_categorical(Y_train, nb_classes)
    return X_train, Y_train

X_train, Y_train = reshape_and_normalize_haralick(haralick_features_list, haralick_labels, 2)


pca, pca_fit, scores_pca = fit_PCA(X_train, n_components=10)
scree_plot(pca_fit)

pca, pca_fit, scores_pca = fit_PCA(X_train, n_components=2)
PCA_components = pd.DataFrame(scores_pca)

elbow_plot(PCA_components)

kmeans, k_means_labels, Y_clust, unique_labels = fit_k_means(PCA_components, Y_train, number_of_clusters=2)

label_count= [[] for i in range(unique_labels)]
for n in range(unique_labels):

  label_count[n] = counter(Y_clust[n])
print(label_count) #Number of items of a certain category in cluster 1

plot_counts(label_count)

plot_scatter(k_means_labels, kmeans, PCA_components)

print('Reached end')

#
