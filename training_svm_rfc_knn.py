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

from utils import reshape_and_normalize_TC, create_training_data_k_means

param_grid={'C':[0.1,1,10,100],'gamma':[0.0001,0.001,0.1,1],'kernel':['rbf','poly']} # params for SVM

path = os.getcwd()
DATADIR = path+'/labeled_data_brain'

# The following methods don't work with the DATADIR as a string, therefore we convert it to a path using Path from pathlib
DATADIR = Path(DATADIR)

#Sub directories for different categories
CATEGORIES = ["10_1","10_2","10_3"]

print('Path:', path)
print('Data directory:', DATADIR)

valaccs = [] # array to store classification accuracies in

for i in range(0, 10):
  X, Y = create_training_data_k_means()
  X_train, Y_train, x_test, y_test = reshape_and_normalize_TC(X, Y, nb_classes=3)
  X_train = X_train.reshape(120,1*200*200)
  x_test = x_test.reshape(31,1*200*200)
  svc=svm.SVC(probability=True)
  model=GridSearchCV(svc,param_grid)
  model.fit(X_train, Y_train)
  y_pred=model.predict(x_test)
  valaccs.append(metrics.accuracy_score(y_test, y_pred))
print('Validation accuracies:', valaccs,  file=open('traditional_clf_outputs/valacc_SVM_output.txt', "w"))

valaccs = []

for i in range(0, 10):
  X, Y = create_training_data_k_means()
  X_train, Y_train, x_test, y_test = reshape_and_normalize_TC(X, Y, nb_classes=3)
  X_train = X_train.reshape(120,1*200*200)
  x_test = x_test.reshape(31,1*200*200)
  model=RandomForestClassifier(n_estimators=100, random_state=1234)
  model.fit(X_train,Y_train)
  y_pred=model.predict(x_test)
  valaccs.append(metrics.accuracy_score(y_test, y_pred))
print('Validation accuracies:', valaccs,  file=open('traditional_clf_outputs/valacc_RFC_output.txt', "w"))

valaccs = []
for i in range(0, 10):
  X, Y = create_training_data_k_means()
  X_train, Y_train, x_test, y_test = reshape_and_normalize_TC(X, Y, nb_classes=3)
  X_train = X_train.reshape(120,1*200*200)
  x_test = x_test.reshape(31,1*200*200)
  model = KNeighborsClassifier(n_neighbors=3)
  model.fit(X_train,Y_train)
  y_pred=model.predict(x_test)
  valaccs.append(metrics.accuracy_score(y_test, y_pred))
print('Validation accuracies:', valaccs,  file=open('traditional_clf_outputs/valacc_KNN_output.txt', "w"))