from datetime import date as dt
import os
import gc
import pickle as pkl  # module for serialization
from tensorflow.keras import backend as K
import numpy as np
from matplotlib import pyplot as plt
import tensorflow as tf
import tensorflow.keras as keras
from PIL import Image
import os
from tensorflow.keras.preprocessing.image import load_img
from tensorflow.keras.applications.vgg16 import preprocess_input
from tf_keras_vis.utils.model_modifiers import ReplaceToLinear
from tf_keras_vis.utils.scores import CategoricalScore, BinaryScore
from tf_keras_vis.saliency import Saliency
from tensorflow.keras import backend as K
import pickle as pkl # module for serialization
import math
from colab_utils import *
load = False
resnet = False
today = dt.today()
date = today.strftime("%b-%d-%Y")

exp_name, baseline, cutout, shear, gblur, crop, randcomb, mobius, allcomb_sparse, allcomb_full, resnet, inception = read_args()
print('exp_name '+exp_name)
path = os.path.join(os.getcwd(), 'data_control_treated')

allcomb_path = "final_model/final_model"
model = keras.models.load_model(allcomb_path)

if load:
    split_dict = load_test_set("PATH_TO_TEST_SET")

    X = split_dict['X']
    Y = split_dict['Y']
    X_test = split_dict['X_test']
    y_test = split_dict['y_test']
    print(len(X))
    print(len(Y))
    print(len(X_test))
    print(len(y_test))
    print(
        "ratios of labels in the brain data set are {} {} {}".format(round(Y.count(0) / len(Y), 2),
                                                               round(Y.count(1) / len(Y), 2),
                                                               round(Y.count(2) / len(Y), 2)))
    print("ratios of labels in the test set are {} : {} : {}".format(round(y_test.count(0) / len(y_test), 2),
                                                                     round(y_test.count(1) / len(y_test), 2),
                                                                     round(y_test.count(2) / len(y_test), 2)))

else:

    data = create_data(path, duplicate_channels=False, equalize=True)

    print(len(data))

    data_list = []
    data_list.append(data[0:len(data)])

    X = []
    Y = []
    for i in data_list:
        for feature, label in i:
            X.append(feature)
            Y.append(label)

print(
    "ratios of labels in the data set are {} {} ".format(round(Y.count(0) / len(Y), 2), round(Y.count(1) / len(Y), 2)))

X, X_test, Y, y_test = train_test_split(X, Y, test_size=0.2, stratify=Y)
print("ratios of labels in the test set are {} : {} ".format(round(y_test.count(0) / len(y_test), 2),
                                                                 round(y_test.count(1) / len(y_test), 2)))

if not load:
    split_dict = save_test_set(os.path.join(os.getcwd(), 'saved_test_sets'), exp_name, X, X_test, Y, y_test)

# Kfold CV (k=10)
X_train, y_train, X_val, y_val = kfoldcv(X, Y, k=10)

X_train_aug, y_train_aug, X_val_aug, y_val_aug = augment_data(X_train, y_train, X_val, y_val, baseline, cutout, shear,
                                                              gblur, crop, randcomb, mobius, allcomb_full, resnet, limb=True)

results = {"accuracies": [], "losses": [], "val_accuracies": [],
           "val_losses": [], "test_performance": [], "test_accuracies": [], "test_losses": []}
hyperparams = {"configuration": [], "loss_func": [], "optimizer": [], "learning_rate": [], "lambda": []}

for i in range(0, len(X_train_aug)):

    print("training_model_{}".format(i))
    results, hyperparams = train_model(X_train_aug[i], X_val_aug[i], y_train_aug[i], y_val_aug[i], X_test, y_test, exp_name,
                          results, hyperparams, i, model=model, limb=True, lr=0.00001, lmbd=0.0001) #normally lr 0.00001
    K.clear_session()
    tf.compat.v1.reset_default_graph()
    gc.collect()

print('Results:', results, file=open('results/Results_{}_{}.txt'.format(exp_name, date), "w"))
print('Hyperparams:', hyperparams, file=open('results/Hyperparams{}_{}.txt'.format(exp_name, date), "w"))