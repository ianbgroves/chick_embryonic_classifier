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
from colab_utils import *
load = False
# Setting the dataset path
path = os.path.join(os.getcwd(), 'data_10_early_late')
if load:
  split_dict = load_test_set("PATH_TO_SAVED_TEST_SET")
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
  plt.imshow(X[0])
else:

    data = create_data(path, duplicate_channels=False, equalize=True)

    data_list = []
    data_list.append(data[0:len(data)])

    X = []
    Y = []
    for i in data_list:
      for feature, label in i:
        X.append(feature)
        Y.append(label)

#Sub directories for different categories
CATEGORIES = ["10early","10late"]


# X, Y = aug_data_2(X, Y)

X_train, Y_train = reshape_and_normalize(X, Y, nb_classes=2)

pca, pca_fit, scores_pca = fit_PCA(X_train, n_components=10)
scree_plot(pca)

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
