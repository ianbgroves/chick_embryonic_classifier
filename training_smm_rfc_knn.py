
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

# sklearn
from sklearn import svm
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from tensorflow.keras.utils import to_categorical
from sklearn import metrics
from random import shuffle

from binary_utils import *

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

X, X_test, Y, y_test = train_test_split(X, Y, test_size=0.2, stratify=Y)


# print("ratios of labels in the test set are {} : {} : {}".format(round(y_test.count(0) / len(y_test), 2),
#                                                                  round(y_test.count(1) / len(y_test), 2),
#                                                                  round(y_test.count(2) / len(y_test), 2)))

# X_train, y_train, X_val, y_val = kfoldcv(X, Y, k=10)
#
# X_train_aug, y_train_aug, X_val_aug, y_val_aug = augment_data(X_train, y_train, X_val, y_val, baseline, cutout, shear,
#                                                               gblur, crop, randcomb, mobius, allcomb_sparse, allcomb_full, resnet=False)

# X_train, y_train, X_val, y_val = aug_data_2(X, Y, X_val, y_val, X_val_bool=True)

def reshape(train, val, train_label, val_label):
    np_train = np.array(train) / 255
    np_val = np.array(val) / 255

    # reshaped_train = np_train.reshape(-1, 200, 200, 1)
    # reshaped_val = np_val.reshape(-1, 200, 200, 1)
    print(len(np_train))
    print(len(np_val))

    reshaped_train = np_train.reshape(121,1*200*200)
    reshaped_val = np_val.reshape(31,1*200*200)

    reshaped_train.astype('float32')
    reshaped_val.astype('float32')

    train_label = to_categorical(train_label)
    val_label = to_categorical(val_label, 2)
    return reshaped_train, reshaped_val, train_label, val_label

# X_train_aug, X_val, y_train, y_val = reshape(X_train, y_train, X_val, y_val)
# X_train_aug, X_val, y_train, y_val = reshape(X, X_test, Y, y_test)
valaccs = [] # array to store classification accuracies in

results = {"accuracies": [], "losses": [], "val_accuracies": [],
           "val_losses": [], "test_performance": [], "test_accuracies": [], "test_losses": []}
hyperparams = {"configuration": [], "loss_func": [], "optimizer": [], "learning_rate": [], "lambda": []}

# for i in range(0, len(X_train_aug)):
print("training_model")
# results, hyperparams = train_traditional_cf_model(X, X_test, Y, y_test,
#                                    exp_name,
#                                    results, hyperparams, i, lr=0.00001, lmbd=0.0001)

#
# for i in range(0, 10):
#
#
#   svc=svm.SVC(probability=True)
#   model=GridSearchCV(svc,param_grid)
#   model.fit(X_train, Y_train)
#   y_pred=model.predict(x_test)
#   valaccs.append(metrics.accuracy_score(y_test, y_pred))
# print('Validation accuracies:', valaccs,  file=open('traditional_clf_outputs/valacc_SVM_output.txt', "w"))
#
# valaccs = []
#
# for i in range(0, 10):
#   X, Y = create_training_data_k_means()
#   X_train, Y_train, x_test, y_test = reshape_and_normalize_TC(X, Y, nb_classes=3)
#   X_train = X_train.reshape(120,1*200*200)
#   x_test = x_test.reshape(31,1*200*200)
#   model=RandomForestClassifier(n_estimators=100, random_state=1234)
#   model.fit(X_train,Y_train)
#   y_pred=model.predict(x_test)
#   valaccs.append(metrics.accuracy_score(y_test, y_pred))
# print('Validation accuracies:', valaccs,  file=open('traditional_clf_outputs/valacc_RFC_output.txt', "w"))
#
valaccs = []
for i in range(0, 10):
  print("here")
  # X, Y = create_training_data_k_means()
  # X_train, Y_train, x_test, y_test = reshape_and_normalize_TC(X, Y, nb_classes=3)
  X = np.array(X)
  X_test = np.array(X_test)
  X_train = X.reshape(121,1*200*200)
  x_test = X_test.reshape(31,1*200*200)
  model = KNeighborsClassifier(n_neighbors=3)
  model.fit(X_train,Y)
  y_pred=model.predict(x_test)
  print("there")
  valaccs.append(metrics.accuracy_score(y_test, y_pred))

  data = create_data('data_10a_b', duplicate_channels=False, equalize=False)

  data_list = []
  data_list.append(data[0:len(data)])

  X = []
  Y = []
  for i in data_list:
      for feature, label in i:
          X.append(feature)
          Y.append(label)

  X, X_test, Y, y_test = train_test_split(X, Y, test_size=0.2, stratify=Y)

print('Validation accuracies:', valaccs,  file=open('traditional_clf_outputs/valacc_KNN_output.txt', "w"))