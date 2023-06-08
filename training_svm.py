import seaborn
import numpy as np
import matplotlib.pyplot as plt
import os
import cv2
import time
import random
import tensorflow
import sklearn
import pandas as pd
from pathlib import Path
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm
# sklearn
from sklearn import svm
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from tensorflow.keras.utils import to_categorical
from sklearn import metrics
from random import shuffle

from binary_utils import *

def reshape_and_normalize_TC(X, Y, nb_classes):
    shuffle(X)
    shuffle(Y)
    # X_train, x_test, Y_train, y_test = train_test_split(X, Y, test_size=0.2)  # 70% training and 30% test
    X_train = np.array(X_train).reshape(-1, 200, 200,
                                        1)  # Array containing every pixel as an index (1D array - 40,000 long)
    x_test = np.array(x_test).reshape(-1, 200, 200, 1)
    X_train = np.array(X_train)
    Y_train = np.array(Y_train)
    x_test = np.array(x_test)
    y_test = np.array(y_test)

    X_train = X_train.astype('float32')

    # Normalization

    X_train = X_train / 255.0
    x_test = x_test / 255.0

    # convert class vectors to binary class matrices with one-hot encoding

    return X_train, Y_train, x_test, y_test


param_grid={'C':[0.1,1,10,100],'gamma':[0.0001,0.001,0.1,1],'kernel':['rbf','poly']} # params for SVM

load = False
os.chdir(r'G:\My Drive\Python_projects\classifier\binary_classification')

exp_name, baseline, cutout, shear, gblur, crop, randcomb, mobius, allcomb_sparse, allcomb_full = read_args()
print('exp_name '+exp_name)

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

valaccs = []
X_train, x_test, Y_train, y_test = train_test_split(X, Y, test_size=0.2)  # 70% training and 20% test

X_train, x_test, Y_train, y_test = augment_data_hd_cutout(X_train, Y_train, x_test, y_test, cutout, randcomb)

X_train = np.array(X_train).reshape(-1, 200, 200,
                                    1)  # Array containing every pixel as an index (1D array - 40,000 long)
x_test = np.array(x_test).reshape(-1, 200, 200, 1)
X_train = np.array(X_train)
Y_train = np.array(Y_train)
x_test = np.array(x_test)
y_test = np.array(y_test)

X_train = X_train.astype('float32')

print("X_train shape {}".format(X_train.shape))
print("x_test shape {}".format(x_test.shape))
print("Y_train shape {}".format(Y_train.shape))
print("y_test shape {}".format(y_test.shape))
# Normalization

X_train = X_train / 255.0
x_test = x_test / 255.0

for i in tqdm(range(0, 1)):

  # X_train, Y_train, x_test, y_test = reshape_and_normalize_TC(X_train, y_train, nb_classes=2)
  X_train = X_train.reshape(-1,1*200*200)
  x_test = x_test.reshape(-1,1*200*200)
  print("constructing SVM model")
  svc=svm.SVC(probability=True)
  model=GridSearchCV(svc,param_grid)
  print("fitting SVM model")
  model.fit(X_train, Y_train)
  y_pred=model.predict(x_test)
  valaccs.append(metrics.accuracy_score(y_test, y_pred))
  print("valaccs {}".format(valaccs))

print('Validation accuracies:', valaccs,  file=open('traditional_clf_outputs/valacc_SVM_output_{}.txt'.format(exp_name), "w"))

# valaccs = []
#
# for i in tqdm(range(0,10)):
#     X_train = X_train.reshape(-1, 1 * 200 * 200)
#     x_test = x_test.reshape(-1, 1 * 200 * 200)
#     print("constructing RFC model")
#     model=RandomForestClassifier(n_estimators=100)
#     print("fitting RFC model")
#     model.fit(X_train, Y_train)
#     y_pred = model.predict(x_test)
#     valaccs.append(metrics.accuracy_score(y_test, y_pred))
#     print("valaccs {}".format(valaccs))
#
# print('Validation accuracies:', valaccs,
#       file=open('traditional_clf_outputs/valacc_RFC_output_{}.txt'.format(exp_name), "w"))

# valaccs = []
#
# for i in tqdm(range(0,10)):
#     X_train = X_train.reshape(-1, 1 * 200 * 200)
#     x_test = x_test.reshape(-1, 1 * 200 * 200)
#     print("constructing KNN model")
#     model = KNeighborsClassifier(n_neighbors=3)
#     print("fitting KNN model")
#     model.fit(X_train, Y_train)
#     y_pred = model.predict(x_test)
#     valaccs.append(metrics.accuracy_score(y_test, y_pred))
#     print("valaccs {}".format(valaccs))
#     del model
# print('Validation accuracies:', valaccs,
#       file=open('traditional_clf_outputs/valacc_KNN_output_{}.txt'.format(exp_name), "w"))
