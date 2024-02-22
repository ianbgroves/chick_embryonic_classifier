import os
import gc
import sys
import pickle as pkl  # module for serialization
from pathlib import Path
import numpy as np

from random import random
from random import seed, shuffle
from random import shuffle

from argparse import ArgumentParser
from argparse import ArgumentParser
from datetime import date as dt

import cv2
from PIL import Image, ImageOps
from imgaug import augmenters as iaa

import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.ticker as plticker
import seaborn as sns
sns.set_theme(style="white")

from sklearn.model_selection import LeaveOneOut, KFold
from sklearn.model_selection import train_test_split, GridSearchCV

import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras import backend as k
from tensorflow.keras.utils import to_categorical
from tensorflow.keras import regularizers
from tensorflow.keras.layers import LeakyReLU, Softmax
from tensorflow.keras.layers import Conv2D, MaxPooling2D, SeparableConv2D
from tensorflow.keras.layers import Dense, Flatten, Dropout, Reshape, Activation
from tensorflow.keras.layers import InputLayer, Input
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.layers import LSTM
from tensorflow.keras.optimizers import Adam, Adagrad, Adadelta, Adamax, SGD, Ftrl, Nadam, RMSprop
from tensorflow.keras.callbacks import TensorBoard
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.utils import to_categorical
from tensorflow.keras import backend as K
from tensorflow.keras import regularizers
from tensorflow.keras import Sequential

# Resnet imports
from tensorflow.keras.applications.resnet50 import ResNet50
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.resnet50 import preprocess_input

# inceptionv3 imports
from tensorflow.keras.applications.inception_v3 import InceptionV3
from tensorflow.keras.applications.inception_v3 import preprocess_input as preprocess_inception

# VGG16 imports
from tensorflow.keras.preprocessing.image import load_img
from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras.applications.vgg16 import preprocess_input as preprocess_vgg

def read_args(baseline=False, cutout=False, shear=False, gblur=False, crop=False, randcomb=False, mobius=False,
              allcomb_sparse=False,
              allcomb_full=False, resnet=False, inception=False):
    parser = ArgumentParser(
        description='provide an experiment name and one augmentation, type --help for details/list of augmentations')

    parser.add_argument("-exp", "--exp", dest="expname", type=str,
                        help="write to file named expname")
    parser.add_argument('--baseline', dest='baseline', default=False, action='store_true',
                        help='if set, augments the data with baseline')
    parser.add_argument('--cutout', dest='cutout', default=False, action='store_true',
                        help='if set, augments the data with cutout')
    parser.add_argument('--shear', dest='shear', default=False, action='store_true',
                        help='if set, augments the data with shear')
    parser.add_argument('--gblur', dest='gblur', default=False, action='store_true',
                        help='if set, augments the data with gblur')
    parser.add_argument('--crop', dest='crop', default=False, action='store_true',
                        help='if set, augments the data with crop')
    parser.add_argument('--randcomb', dest='randcomb', default=False, action='store_true',
                        help='if set, augments the data with randcomb')
    parser.add_argument('--mobius', dest='mobius', default=False, action='store_true',
                        help='if set, augments the data with mobius transforms')
    parser.add_argument('--allcomb_sparse', dest='allcomb_sparse', default=False, action='store_true',
                        help='if set, augments the data sparsely with baseline, cutout, shear, gblur transforms')
    parser.add_argument('--allcomb_full', dest='allcomb_full', default=False, action='store_true',
                        help='if set, augments each datum with baseline, cutout, shear, gblur transforms')
    parser.add_argument('--resnet', dest='resnet', default=False, action='store_true',
                        help='if set, re-trains resnet50')
    parser.add_argument('--inception', dest='inception', default=False, action='store_true',
                        help='if set, re-trains inception')
    exp_name = parser.parse_args().expname
    assert type(exp_name) == str, 'exp_name is not a string'
    args = parser.parse_args()

    while (True):
        if vars(args)['baseline']:
            baseline = True
            break
        if vars(args)['cutout']:
            cutout = True
            break
        if vars(args)['shear']:
            shear = True
            break
        if vars(args)['gblur']:
            gblur = True
            break
        if vars(args)['crop']:
            crop = True
            break
        if vars(args)['randcomb']:
            randcomb = True
            break
        if vars(args)['mobius']:
            mobius = True
            break
        if vars(args)['allcomb_sparse']:
            allcomb_sparse = True
            break
        if vars(args)['allcomb_full']:
            allcomb_full = True
            break
        else:
            print('No augmentation set, please parse \"--help", or refer to README.txt')
            exit()

    while (True):
        if vars(args)['resnet']:
            resnet = True
            break
        elif vars(args)['inception']:
            inception = True
            break
        else:
            break
    return exp_name, baseline, cutout, shear, gblur, crop, randcomb, mobius, allcomb_sparse, allcomb_full, resnet, inception


def load_test_set(split_save_path):
    with open(split_save_path, 'rb') as pickle_file:
        split_dict = pkl.load(pickle_file)

    return split_dict


def load_data(test_set_path):
    split_dict = load_test_set(test_set_path)
    print(split_dict.keys())
    X = split_dict['X']
    Y = split_dict['Y']
    X_test = split_dict['X_test']
    y_test = split_dict['y_test']

    print(f"Data sizes: X: {len(X)}, Y: {len(Y)}, test_images: {len(X_test)}, test_labels: {len(y_test)}")

    print(
        "ratios of labels in the data set are {} {}".format(round(Y.count(0) / len(Y), 2),
                                                            round(Y.count(1) / len(Y), 2)))
    print(
        "ratios of labels in the test set are {} : {}".format(round(y_test.count(0) / len(y_test), 2),
                                                              round(y_test.count(1) / len(y_test), 2)))
    return X, Y, X_test, y_test


def save_test_set(path, exp_name, X, X_test, Y, y_test):
    today = dt.today()
    date = today.strftime("%b-%d-%Y")

    os.makedirs(path + '/{}_{}'.format(exp_name, date), exist_ok=True)
    split_save_path = (path + '/{}_{}/pkl_splits'.format(exp_name, date))
    to_save = dict(zip(['X', 'X_test', 'Y', 'y_test'], [X, X_test, Y, y_test]))
    with open(split_save_path, 'wb') as pickle_file:
        pkl.dump(to_save, pickle_file)
    with open(split_save_path, 'rb') as pickle_file:
        split_dict = pkl.load(pickle_file)
    assert y_test == split_dict['y_test'], "loaded y_test not the same as Y_test just generated"

    return split_dict


def read_imgs(path, duplicate_channels, equalize=True):
    contents = os.listdir(path)
    data = []
    class_dirs = [path + f'/{contents[0]}', path + f'/{contents[1]}']
    print(class_dirs)
    for img in os.listdir(class_dirs[0]):  # iterate over each image
        try:
            img_array = Image.open(class_dirs[0] + '/{}'.format(img)).convert('L')
            if equalize:
                img_array = ImageOps.equalize(img_array, mask=None)
            if duplicate_channels:
                # img_array = img_array.convert('RGB')
                img_array = Image.open(class_dirs[0] + '/{}'.format(img)).convert('RGB')
            img_array = img_array.resize((200, 200), Image.ANTIALIAS)
            img_array = np.array(img_array)
            data.append([img_array, 0])
        except Exception as e:
            pass

    for img in os.listdir(class_dirs[1]):  # iterate over each image
        try:
            img_array = Image.open(class_dirs[1] + '/{}'.format(img)).convert('L')
            if equalize:
                img_array = ImageOps.equalize(img_array, mask=None)
            if duplicate_channels:
                # img_array = img_array.convert('RGB')
                img_array = Image.open(class_dirs[1] + '/{}'.format(img)).convert('RGB')
            img_array = img_array.resize((200, 200), Image.ANTIALIAS)
            img_array = np.array(img_array)
            data.append([img_array, 1])
        except Exception as e:
            pass

    return data


def create_data_save_test_set(path, mobius, resnet, inception, exp_name):
    if mobius or resnet or inception:
        data = read_imgs(path, duplicate_channels=True, equalize=True)
    else:
        data = read_imgs(path, duplicate_channels=False, equalize=True)

    print(len(data))
    # should be 152

    data_list = []
    data_list.append(data[0:len(data)])

    X = []
    Y = []

    for i in data_list:
        for feature, label in i:
            X.append(feature)
            Y.append(label)

    print(
        "ratios of labels in the data set are {} : {}".format(round(Y.count(0) / len(Y), 2),
                                                              round(Y.count(1) / len(Y), 2),
                                                              round(Y.count(2) / len(Y), 2)))

    print(f"Data sizes: X: {len(X)}, Y: {len(Y)}")
    X, X_test, Y, y_test = train_test_split(X, Y, test_size=0.2, stratify=Y)
    print("ratios of labels in the test set are {} : {} ".format(round(y_test.count(0) / len(y_test), 2),
                                                                 round(y_test.count(1) / len(y_test), 2)))

    split_dict = save_test_set(os.path.join(os.getcwd(), 'saved_test_sets'), exp_name, X, X_test, Y, y_test)
    return X, X_test, Y, y_test


def convert_RGB(X, X_test):
    print("converting to RGB")
    for i in range(0, len(X)):
        X[i] = Image.fromarray(X[i])
        X[i] = X[i].convert("RGB")
        X[i] = np.array(X[i])
        print("x shape is {}".format(X[i].shape))
    for i in range(0, len(X_test)):
        X_test[i] = Image.fromarray(X_test[i])
        X_test[i] = X_test[i].convert("RGB")
        X_test[i] = np.array(X_test[i])
        print("x test shape is {}".format(X_test[i].shape))
    return X, X_test


def kfoldcv(X, Y, k):
    kfold = KFold(n_splits=k)
    kfold.get_n_splits(X)

    X_train = []
    X_val = []
    y_train = []
    y_val = []

    for train_index, test_index in kfold.split(X):
        xvaltemp = []
        yvaltemp = []
        xtraintemp = []
        ytraintemp = []
        for i in range(0, len(test_index)):
            xvaltemp.append(X[test_index[i]])
            yvaltemp.append(Y[test_index[i]])
        X_val.append(xvaltemp)
        y_val.append(yvaltemp)
        for i in range(0, len(train_index)):
            xtraintemp.append(X[train_index[i]])
            ytraintemp.append(Y[train_index[i]])
        X_train.append(xtraintemp)
        y_train.append(ytraintemp)
    return X_train, y_train, X_val, y_val


def augment_data(X_train, y_train, X_val, y_val, baseline=False, cutout=False, shear=False, gblur=False, crop=False,
                 randcomb=False, mobius=False, allcomb_sparse=False, allcomb_full=False, resnet=False, inception=False,
                 limb=False):
    print("resnet is " + str(resnet))
    print("inception is " + str(inception))
    X_val_aug = []
    X_train_aug = []
    y_train_aug = []
    y_val_aug = []
    X_train = np.array(X_train, dtype=object)
    X_val = np.array(X_val, dtype=object)

    # Mode can be chosen from 'reflect’, ‘constant’, ‘nearest’, ‘mirror’, ‘wrap’
    mode = 'constant'

    # if user_defined == False, then it is random (recommended)
    user_defined = False
    start_points = 32, 16, 16, 32, 32, 48
    end_points = 16, 32, 32, 48, 48, 32

    for i in range(0, len(X_train)):  # for every 149 long list of images 0 -> 10
        print(f"augmenting for fold {i}")
        rotated_features = []  # empty temp list of rotated images
        rotated_labels = []  # empty temp list of labels
        X_train[i] = np.array(X_train[i])  # convert list of images to np.array - seems to be necessary for imgaug

        for j in range(0, len(X_train[
                                  i])):  # for every image in 149 long list of images 0 -> 149 #uncomment for full length (very slow)

            # for j in range(0, 3):
            if mobius:
                # M must be >1
                # The smaller M is, the more "normal" the output looks
                M = np.linspace(1.1, 1.2, 20)
                M = np.random.choice(M)
                print(M)

                img = X_train[i][j]
                feature, uninterpolated_image = mobius_fast_interpolation('example', True, img,
                                                                          M,
                                                                          mode=mode, rgb=False,
                                                                          output_height=200,
                                                                          output_width=200,
                                                                          user_defined=user_defined,
                                                                          start_points=start_points,
                                                                          end_points=end_points)
                feature = np.array(feature)
                # print("shape of mobius returned image is {}".format(np.shape(feature)))
                # print("adding feature of len {}".format(np.shape(feature)))
                rotated_features.append(feature)
                # print("adding label {}".format(y_train[i][j]))
                rotated_labels.append(y_train[i][j])

            for angle in np.arange(0, 360, 10):

                if cutout:
                    cutout = iaa.Cutout(nb_iterations=(1, 3),
                                        size=0.2)  #
                    feature = cutout(image=X_train[i][j])

                if shear:
                    shear = iaa.ShearX((-20, 20))
                    feature = shear(image=X_train[i][j])

                if gblur:
                    gblur = iaa.GaussianBlur(sigma=(0.0, 5.0))  # sigma = 5 was originally used - not random
                    feature = gblur(image=X_train[i][j])

                if crop:
                    crop = iaa.Crop(percent=(0.0, 0.3))  # originally every img was cropped 30 times
                    feature = crop(image=X_train[i][j])

                if randcomb:
                    seq = iaa.SomeOf(1, [iaa.Cutout(nb_iterations=(1, 3)), iaa.GaussianBlur(sigma=(0.0, 3.0)),
                                         iaa.ShearX((-20, 20))])  # Just one
                    feature = seq(image=X_train[i][j])

                # if allcomb_sparse:
                #     seq = iaa.Sequential([iaa.Cutout(nb_iterations=(1, 3)), iaa.GaussianBlur(sigma=(0.0, 3.0)),
                #                          iaa.ShearX((-20, 20))], random_order=True)
                #     feature = seq(image=X_train[i][j])

                if allcomb_full:
                    cutout = iaa.Cutout(nb_iterations=(1, 3),
                                        size=0.2)
                    cutout_feature = cutout(image=X_train[i][j])

                    shear = iaa.ShearX((-20, 20))
                    shear_feature = shear(image=X_train[i][j])

                    gblur = iaa.GaussianBlur(sigma=(2.5, 5.0))  # sigma = 5 was originally used - not random
                    gblur_feature = gblur(image=X_train[i][j])

                    rotated_features.append(cutout_feature)
                    rotated_labels.append(y_train[i][j])

                    rotated_features.append(shear_feature)
                    rotated_labels.append(y_train[i][j])

                    rotated_features.append(gblur_feature)
                    rotated_labels.append(y_train[i][j])

                if limb:
                    flip = iaa.Fliplr(1.0)
                    flip_feature = flip(image=X_train[i][j])

                    rotated_features.append(flip_feature)
                    rotated_labels.append(y_train[i][j])

                rotate = iaa.geometric.Affine(rotate=angle)  # set up augmenter
                feature = rotate(image=X_train[i][j])  # rotate each image 36 times
                feature = Image.fromarray(feature)
                if not resnet and not inception:
                    feature = feature.convert("L")
                feature = np.array(feature)
                rotated_features.append(feature)  # append to 36 rotated images to temp
                rotated_labels.append(y_train[i][j])

        X_train_aug.append(
            rotated_features)  # when finished all j append 149 long list of 36 long list of rotated imgs, proceed to next i
        y_train_aug.append(rotated_labels)

        print("number of training images post augmentation {}".format(len(X_train_aug[0])))

        if mobius:
            assert len(X_train_aug[0]) == (len(X_train[0]) * len(
                np.arange(0, 360,
                          10)) + len(
                X_train[0])), "X_train_aug is not equal to X_train multiplied by the number of transformations"

        if allcomb_full:
            if limb:
                assert len(X_train_aug[0]) == (len(X_train[0]) * len(
                    np.arange(0, 360,
                              10))) * 5, "X_train_aug is not equal to X_train multiplied by the number of transformations"
            else:

                assert len(X_train_aug[0]) == len(X_train[0]) * (len(np.arange(0, 360,
                                                                               10)) * 4), "X_train_aug is not equal to X_train multiplied by the number of transformations"

        if limb and (not allcomb_full):
            assert len(X_train_aug[0]) == (len(X_train[0]) * len(
                np.arange(0, 360,
                          10))) * 2, "X_train_aug is not equal to X_train multiplied by the number of transformations"

        elif not allcomb_full:

            assert len(X_train_aug[0]) == (len(X_train[0]) * len(
                np.arange(0, 360,
                          10))), "X_train_aug is not equal to X_train multiplied by the number of transformations"

    for i in range(0, len(X_val)):  # for every 149 long list of images 0 -> 10

        rotated_features = []  # empty temp list of rotated images
        rotated_labels = []  # empty temp list of labels
        X_val[i] = np.array(X_val[i])  # convert list of images to np.array - seems to be necessary for imgaug

        for j in range(0, len(X_val[i])):  # for every image in 149 long list of images 0 -> 149

            if mobius:
                # M must be >1
                # The smaller M is, the more "normal" the output looks
                M = np.linspace(1.1, 1.2, 20)
                M = np.random.choice(M)
                print(M)

                img = X_val[i][j]
                feature, uninterpolated_image = mobius_fast_interpolation('example', True, img,
                                                                          M,
                                                                          mode=mode, rgb=False,
                                                                          output_height=200,
                                                                          output_width=200,
                                                                          user_defined=user_defined,
                                                                          start_points=start_points,
                                                                          end_points=end_points)
                feature = np.array(feature)
                # print("shape of mobius returned val image is {}".format(np.shape(feature)))
                rotated_features.append(feature)
                rotated_labels.append(y_val[i][j])

            for angle in np.arange(0, 360, 10):

                if cutout:
                    cutout = iaa.Cutout(nb_iterations=(1, 3),
                                        size=0.2)  #
                    feature = cutout(image=X_val[i][j])
                if shear:
                    shear = iaa.ShearX((-20, 20))
                    feature = shear(image=X_val[i][j])
                if gblur:
                    gblur = iaa.GaussianBlur(sigma=(0.0, 5.0))  # sigma = 5 was originally used - not random
                    feature = gblur(image=X_val[i][j])
                if crop:
                    crop = iaa.Crop(percent=(0.0, 0.3))  # originally every img was cropped 30 times
                    feature = crop(image=X_val[i][j])
                if randcomb:
                    seq = iaa.SomeOf(1, [iaa.Cutout(nb_iterations=(1, 3)), iaa.GaussianBlur(sigma=(0.0, 3.0)),
                                         iaa.ShearX((-20, 20))])  # Just one
                    feature = seq(image=X_val[i][j])

                if allcomb_full:
                    cutout = iaa.Cutout(nb_iterations=(1, 3),
                                        size=0.2)  #
                    cutout_feature = cutout(image=X_val[i][j])

                    shear = iaa.ShearX((-20, 20))
                    shear_feature = shear(image=X_val[i][j])

                    gblur = iaa.GaussianBlur(sigma=(2.5, 5.0))  # sigma = 5 was originally used - not random
                    gblur_feature = gblur(image=X_val[i][j])

                    rotated_features.append(cutout_feature)
                    rotated_labels.append(y_val[i][j])

                    rotated_features.append(shear_feature)
                    rotated_labels.append(y_val[i][j])

                    rotated_features.append(gblur_feature)
                    rotated_labels.append(y_val[i][j])

                if limb:
                    flip = iaa.Fliplr(0.5)
                    flip_feature = flip(image=X_val[i][j])

                    rotated_features.append(flip_feature)
                    rotated_labels.append(y_val[i][j])

                rotate = iaa.geometric.Affine(rotate=angle)  # set up augmenter
                feature = rotate(image=X_val[i][j])  # rotate each image 36 times
                feature = Image.fromarray(feature)

                if not resnet and not inception:
                    feature = feature.convert("L")
                feature = np.array(feature)

                rotated_features.append(feature)  # append to 36 rotated images to temp
                rotated_labels.append(y_val[i][j])

        X_val_aug.append(
            rotated_features)  # when finished all j append 149 long list of 36 long list of rotated imgs, proceed to next i
        y_val_aug.append(rotated_labels)

        if mobius:
            assert len(X_val_aug[0]) == (len(X_val[0]) * len(
                np.arange(0, 360,
                          10)) + len(
                X_val[
                    0])), "X_val_aug is not equal to X_train multiplied by the number of transformations plus a mobius transformation of each img"

        if limb and not allcomb_full:
            # print("X_val_aug len is {}".format(len(X_val_aug[0])))
            assert len(X_val_aug[0]) == (len(X_val[0]) * len(
                np.arange(0, 360,
                          10))) * 2, "X_val_aug is not equal to X_val multiplied by the number of transformations"

        if allcomb_full:
            if limb:

                assert len(X_val_aug[0]) == (len(X_val[0]) * len(
                    np.arange(0, 360,
                              10))) * 5, "X_train_aug is not equal to X_train multiplied by the number of transformations"
            else:

                assert len(X_val_aug[0]) == (len(X_val[0]) * len(
                    np.arange(0, 360,
                              10)) * 4), "X_train_aug is not equal to X_train multiplied by the number of transformations"
        if limb and (not allcomb_full):
            assert len(X_val_aug[0]) == (len(X_val[0]) * len(
                np.arange(0, 360,
                          10))) * 2, "X_train_aug is not equal to X_train multiplied by the number of transformations"
        else:
            # print("X_val_aug len is {}".format(len(X_val_aug[0])))
            assert len(X_val_aug[0]) == (len(X_val[0]) * len(
                np.arange(0, 360,
                          10))), "X_val_aug is not equal to X_val multiplied by the number of transformations"

    return X_train_aug, y_train_aug, X_val_aug, y_val_aug


def train_model_resnet50(train, val, train_label, val_label, X_test, Y_test, name, results, hyperparams, i, model=None,
                         pretrained=False, freeze=False):
    np_train = np.array(train) / 255
    np_val = np.array(val) / 255

    # reshaped_train = np_train.reshape(-1, 200, 200, 1)
    # reshaped_val = np_val.reshape(-1, 200, 200, 1)

    # reshaped_train.astype('float32')
    # reshaped_val.astype('float32')

    print("train shape before preprocess {}".format(np_train.shape))
    reshaped_train = preprocess_input(np_train)
    reshaped_val = preprocess_input(np_val)

    train_label = to_categorical(train_label)
    val_label = to_categorical(val_label, 2)

    print('reshaped_train shape:', np.shape(reshaped_train))
    print('reshaped_val shape:', np.shape(reshaped_val))
    print('train_label shape:', np.shape(train_label))
    print('val_label shape:', np.shape(val_label))

    if pretrained:
        model.compile(optimizer=Adam(learning_rate=0.00001),
                      loss='categorical_crossentropy',
                      metrics=['accuracy'])
        print('using pretrained model')
    else:
        print('building new resnet50 model')
        resnet50_model = ResNet50(include_top=False, weights='imagenet', input_tensor=None, input_shape=(200, 200, 3),
                                  pooling=None)
        if freeze:
            print("freezing base ResNet layers")
            num_layers_to_freeze = 10
            for layer in resnet50_model.layers[:num_layers_to_freeze]:
                layer.trainable = False

        flattened_output = tf.keras.layers.Flatten()(resnet50_model.output)
        fc_classification_layer = tf.keras.layers.Dense(2, activation='softmax')(flattened_output)
        model = tf.keras.models.Model(inputs=resnet50_model.input, outputs=fc_classification_layer)
        model.compile(optimizer=Adam(learning_rate=0.000001),
                      loss='binary_crossentropy',
                      metrics=['accuracy'])

    earlyStop = EarlyStopping(monitor='val_loss', min_delta=0.001, patience=100, mode='auto', restore_best_weights=True)
    epochs = 500
    history = model.fit(x=reshaped_train,
                        y=train_label,
                        epochs=epochs,
                        batch_size=32,
                        validation_data=(reshaped_val, val_label), verbose=2, callbacks=[earlyStop])

    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    accuracy = history.history['val_accuracy'][-1]

    results["accuracies"].append((round(max(acc) * 100, 1)))
    results["losses"].append((round(min(loss) * 100, 1)))
    results["val_accuracies"].append((round(max(val_acc) * 100, 1)))
    results["val_losses"].append((round(min(val_loss) * 100, 1)))
    results["test_accuracies"].append((round(max(acc) * 100, 1)))

    today = dt.today()
    date = today.strftime("%b-%d-%Y")
    model.save(name + '{}'.format(date))

    X_test = np.array(X_test) / 255
    X_test = X_test.reshape(X_test.shape[0], 200, 200, 3)
    Y_test = to_categorical(Y_test)

    print('X_test shape:', np.shape(X_test))  # should be (number of test images, 200, 200, 1)
    print('y_test shape', np.shape(Y_test))  # should be (number of test labels, 3)

    results["test_performance"].append(model.evaluate(X_test, Y_test))
    results["test_accuracies"].append(model.evaluate(X_test, Y_test)[0])
    del model
    K.clear_session()

    return results


def train_model_inception(train, val, train_label, val_label, X_test, Y_test, name, results, hyperparams, i, model=None,
                          pretrained=False):
    np_train = np.array(train) / 255
    np_val = np.array(val) / 255

    print("train shape before preprocess {}".format(np_train.shape))
    reshaped_train = preprocess_input(np_train)
    reshaped_val = preprocess_input(np_val)

    train_label = to_categorical(train_label)
    val_label = to_categorical(val_label, 2)

    print('reshaped_train shape:', np.shape(reshaped_train))
    print('reshaped_val shape:', np.shape(reshaped_val))
    print('train_label shape:', np.shape(train_label))
    print('val_label shape:', np.shape(val_label))

    if pretrained:
        model.compile(optimizer=Adam(learning_rate=0.00001),
                      loss='categorical_crossentropy',
                      metrics=['accuracy'])
        print('using pretrained model')
    else:
        print('building new inception model')
        inception_model = InceptionV3(include_top=False, weights='imagenet', input_tensor=None,
                                      input_shape=(200, 200, 3),
                                      pooling=None)
        flattened_output = tf.keras.layers.Flatten()(inception_model.output)
        fc_classification_layer = tf.keras.layers.Dense(2, activation='softmax')(flattened_output)
        model = tf.keras.models.Model(inputs=inception_model.input, outputs=fc_classification_layer)
        model.compile(optimizer=Adam(learning_rate=0.000001),
                      loss='binary_crossentropy',
                      metrics=['accuracy'])

    earlyStop = EarlyStopping(monitor='val_loss', min_delta=0.001, patience=100, mode='auto', restore_best_weights=True)
    epochs = 500
    history = model.fit(x=reshaped_train,
                        y=train_label,
                        epochs=epochs,
                        batch_size=32,
                        validation_data=(reshaped_val, val_label), verbose=2, callbacks=[earlyStop])

    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    accuracy = history.history['val_accuracy'][-1]

    results["accuracies"].append((round(max(acc) * 100, 1)))
    results["losses"].append((round(min(loss) * 100, 1)))
    results["val_accuracies"].append((round(max(val_acc) * 100, 1)))
    results["val_losses"].append((round(min(val_loss) * 100, 1)))
    results["test_accuracies"].append((round(max(acc) * 100, 1)))

    today = dt.today()
    date = today.strftime("%b-%d-%Y")
    model.save(name + '{}'.format(date))

    X_test = np.array(X_test) / 255
    X_test = X_test.reshape(X_test.shape[0], 200, 200, 3)
    Y_test = to_categorical(Y_test)

    print('X_test shape:', np.shape(X_test))  # should be (number of test images, 200, 200, 1)
    print('y_test shape', np.shape(Y_test))  # should be (number of test labels, 3)

    results["test_performance"].append(model.evaluate(X_test, Y_test))
    results["test_accuracies"].append(model.evaluate(X_test, Y_test)[0])
    del model
    K.clear_session()

    return results


def finetune_resnet_inception(X_train_aug, X_val_aug, y_train_aug, y_val_aug, X_test, y_test,
                              exp_name, results, hyperparams, model, pretrained=True, freeze=True, resnet=True,
                              inception=False):
    if resnet and not inception:
        for i in range(0, len(X_train_aug)):
            K.clear_session()
            tf.compat.v1.reset_default_graph()
            gc.collect()

            print("training_resnet50_model_{}".format(i))
            print("train shape before sending to resnet {}".format(np.array(X_train_aug[i]).shape))
            results = train_model_resnet50(X_train_aug[i], X_val_aug[i], y_train_aug[i], y_val_aug[i], X_test, y_test,
                                           exp_name, results, hyperparams, i, model=None, pretrained=False, freeze=True)
            k.clear_session()
            tf.compat.v1.reset_default_graph()
            gc.collect()

    elif inception and not resnet:
        for i in range(0, len(X_train_aug)):
            K.clear_session()
            tf.compat.v1.reset_default_graph()
            gc.collect()

            print("training_InceptionV3_model_{}".format(i))
            print("train shape before sending to inception {}".format(np.array(X_train_aug[i]).shape))
            results = train_model_inception(X_train_aug[i], X_val_aug[i], y_train_aug[i], y_val_aug[i], X_test, y_test,
                                            exp_name, results, hyperparams, i, model=None, pretrained=False)
            k.clear_session()
            tf.compat.v1.reset_default_graph()
            gc.collect()
    return results, hyperparams


def train_model(train, val, train_label, val_label, X_test, Y_test, name, results, hyperparams, i, model=None,
                limb=False, lr=0.00001, lmbd=0.0001):
    print("type of train {}".format(type(train)))
    print("type of val {}".format(type(val)))
    print("len of train {}".format(len(train)))
    print("len of val {}".format(len(val)))
    print("len of train_i {}".format(len(train[0])))
    print("len of val_i {}".format(len(val[0])))

    np_train = np.array(train) / 255
    np_val = np.array(val) / 255

    reshaped_train = np_train.reshape(-1, 200, 200, 1)
    reshaped_val = np_val.reshape(-1, 200, 200, 1)

    reshaped_train.astype('float32')
    reshaped_val.astype('float32')

    train_label = to_categorical(train_label)
    val_label = to_categorical(val_label, 2)

    print('reshaped_train shape:', np.shape(reshaped_train))
    print('reshaped_val shape:', np.shape(reshaped_val))
    print('train_label shape:', np.shape(train_label))
    print('val_label shape:', np.shape(val_label))

    layer_drop = 0.2
    final_drop = 0.5
    activation = 'relu'

    if model is None:
        model = Sequential([

            layers.Conv2D(16, 3, padding='same', activation=activation, input_shape=(200, 200, 1),
                          kernel_regularizer=regularizers.l2(lmbd)),
            layers.Dropout(layer_drop),
            layers.MaxPooling2D((2, 2), padding='same'),
            layers.Conv2D(32, 3, padding='same', activation=activation, kernel_regularizer=regularizers.l2(lmbd)),
            layers.Dropout(layer_drop),
            layers.MaxPooling2D((2, 2), padding='same'),
            layers.Conv2D(64, 3, padding='same', activation=activation, kernel_regularizer=regularizers.l2(lmbd)),
            layers.Dropout(layer_drop),
            layers.MaxPooling2D((2, 2), padding='same'),
            layers.Conv2D(128, 3, padding='same', activation=activation, kernel_regularizer=regularizers.l2(lmbd)),
            layers.Dropout(layer_drop),
            layers.MaxPooling2D((2, 2), padding='same'),
            layers.Conv2D(256, 3, padding='same', activation=activation, kernel_regularizer=regularizers.l2(lmbd)),
            layers.Dropout(layer_drop),
            layers.MaxPooling2D((2, 2), padding='same'),
            layers.Conv2D(512, 3, padding='same', activation=activation, kernel_regularizer=regularizers.l2(lmbd)),
            layers.Dropout(layer_drop),
            layers.MaxPooling2D((2, 2), padding='same'),
            layers.Conv2D(1024, 3, padding='same', activation=activation, kernel_regularizer=regularizers.l2(lmbd)),
            layers.Dropout(layer_drop),
            layers.MaxPooling2D((2, 2), padding='same'),
            layers.Flatten(),
            layers.Dense(1024, activation=activation),
            layers.Dense(2048, activation=activation),
            layers.Dense(2048, activation=activation),
            layers.Dropout(final_drop),
            layers.Dense(2, activation='softmax')

        ])

    if limb:

        for layer in model.layers:
            if layer.name not in ['dense_3', 'dense_2', 'dense_1']:
                layer.trainable = False

            if layer.trainable:
                print("trainable layer = {}".format(layer.name))

    model.summary()
    opt = Adam(learning_rate=lr)
    model.compile(optimizer=opt,
                  loss='binary_crossentropy',
                  metrics=['accuracy'])

    earlyStop = EarlyStopping(monitor='val_loss', min_delta=0.001, patience=20, mode='auto', restore_best_weights=True)
    epochs = 250

    history = model.fit(x=reshaped_train,
                        y=train_label,
                        epochs=epochs,
                        batch_size=32,
                        validation_data=(reshaped_val, val_label), verbose=2, callbacks=[earlyStop])

    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    accuracy = history.history['val_accuracy'][-1]

    results["accuracies"].append((round(max(acc) * 100, 1)))
    results["losses"].append((round(min(loss) * 100, 1)))
    results["val_accuracies"].append((round(max(val_acc) * 100, 1)))
    results["val_losses"].append((round(min(val_loss) * 100, 1)))

    hyperparams["configuration"].append(model.get_config())
    hyperparams["loss_func"].append(model.loss)
    hyperparams["optimizer"].append(model.optimizer)
    hyperparams["learning_rate"].append(lr)
    hyperparams["lambda"].append(lmbd)

    today = dt.today()
    date = today.strftime("%b-%d-%Y")
    model.save(os.path.join("saved_models/", name,
                            '_{}_{}'.format(date, i)))

    X_test = np.array(X_test) / 255
    X_test = X_test.reshape(X_test.shape[0], 200, 200, 1)
    Y_test = to_categorical(Y_test)

    print('X_test shape:', np.shape(X_test))  # should be (number of test images, 200, 200, 1)
    print('y_test shape', np.shape(Y_test))  # should be (number of test labels, 2)

    results["test_accuracies"].append(model.evaluate(X_test, Y_test)[1])
    results["test_losses"].append(model.evaluate(X_test, Y_test)[0])

    del model

    return results, hyperparams


if __name__ == '__main__':

    today = dt.today()
    date = today.strftime("%b-%d-%Y")
    load = False
    exp_name, baseline, cutout, shear, gblur, crop, randcomb, mobius, allcomb_sparse, allcomb_full, resnet, inception = read_args()
    print('exp_name ' + exp_name)
    path = os.path.join(os.getcwd(), 'data_10_early_late')
    saved_test_path = "saved_test_sets/binary_baseline_3_Mar-14-2023/pkl_splits"

    if load:
        X, Y, X_test, y_test = load_data(saved_test_path)

    else:
        X, X_test, Y, y_test = create_data_save_test_set(path, mobius, resnet, inception, exp_name)

    if mobius or resnet or inception:
        X, X_test = convert_RGB(X, X_test)

    # Kfold CV (k=10). Splits the data into k non-overlapping folds.
    X_train, y_train, X_val, y_val = kfoldcv(X, Y, k=10)

    # Augment data with chosen augmentation regime.
    X_train_aug, y_train_aug, X_val_aug, y_val_aug = augment_data(X_train, y_train, X_val, y_val, baseline, cutout,
                                                                  shear,
                                                                  gblur, crop, randcomb, mobius, allcomb_sparse,
                                                                  allcomb_full, resnet, inception, limb=False)

    # Set up dictionaries for logging.
    results = {"accuracies": [], "losses": [], "val_accuracies": [],
               "val_losses": [], "test_performance": [], "test_accuracies": [], "test_losses": []}
    hyperparams = {"configuration": [], "loss_func": [], "optimizer": [], "learning_rate": [], "lambda": []}

    # In the paper we finetuned ResNet50/InceptionV3
    # From scratch-training takes a long time and careful management of the learning rate,
    # pass either your already trained model or set pretrained=True

    if resnet or inception:
        results, hyperparams = finetune_resnet_inception(X_train_aug, X_val_aug, y_train_aug, y_val_aug, X_test, y_test,
                                                         exp_name, results, hyperparams, model=None, pretrained=True,
                                                         freeze=True, resnet=True, inception=False)

    # If not using Resnet/Inception, 'our model' in the paper.

    else:

        for i in range(0, len(X_train_aug)):
            print("training_our_model_{}".format(i))
            results, hyperparams = train_model(X_train_aug[i], X_val_aug[i], y_train_aug[i], y_val_aug[i], X_test,
                                               y_test,
                                               exp_name,
                                               results, hyperparams, i, lr=0.00001, lmbd=0.0001)
            k.clear_session()
            tf.compat.v1.reset_default_graph()
            gc.collect()

    print('Results:', results, file=open('results/Results_{}_{}.txt'.format(exp_name, date), "w"))
    print('Hyperparams:', hyperparams, file=open('results/Hyperparams{}_{}.txt'.format(exp_name, date), "w"))
