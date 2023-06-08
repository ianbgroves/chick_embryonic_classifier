import os, sys
import cv2
from pathlib import Path
from random import seed, shuffle
import numpy as np
# from numba import cuda
from imgaug import augmenters as iaa
import joblib
from sklearn.model_selection import train_test_split
from PIL import Image, ImageOps
from tensorflow.keras.preprocessing.image import load_img
from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras.applications.vgg16 import preprocess_input as preprocess_vgg
from sklearn.model_selection import LeaveOneOut, KFold
import matplotlib.pyplot as plt
from datetime import date as dt
import pickle as pkl  # module for serialization
import matplotlib.pyplot as plt
# resnet50 imports
from tensorflow.keras.applications.resnet50 import ResNet50
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.resnet50 import preprocess_input
import matplotlib.ticker as plticker
import matplotlib as mpl
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

from sklearn import svm
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from tensorflow.keras.utils import to_categorical
from sklearn import metrics
from random import shuffle

# inceptionv3 imports
from tensorflow.keras.applications.inception_v3 import InceptionV3
from tensorflow.keras.applications.inception_v3 import preprocess_input as preprocess_inception

from argparse import ArgumentParser
import seaborn as sns
sns.set_theme(style="white")

def read_args(baseline=False, cutout=False, shear=False, gblur=False, crop=False, randcomb=False, mobius=False, allcomb_sparse=False, allcomb_full=False):
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
    print(randcomb)
    return exp_name, baseline, cutout, shear, gblur, crop, randcomb, mobius, allcomb_sparse, allcomb_full


def scree_plot(pca):
    import matplotlib.ticker as plticker

    xloc = plticker.MultipleLocator(base=1.0)  # this locator puts ticks at regular intervals
    yloc = plticker.MultipleLocator(base=100000)  # this locator puts ticks at regular intervals

    fig, ax = plt.subplots(figsize=(5,5))
    PC_values = np.arange(pca.n_components_) + 1
    ax.plot(PC_values, pca.explained_variance_ratio_, color='red', linewidth=2.0)
    ax.xaxis.set_major_locator(xloc)

    ax.spines['top'].set_lw(2)
    ax.spines['bottom'].set_lw(2)
    ax.spines['left'].set_lw(2)
    ax.spines['right'].set_lw(2)
    # ax.set_aspect('equal')    # ax.set_ylim(0, (round(max(pca.explained_variance_ratio_))))
    ax.tick_params(axis='both', which='major', labelsize=16)
    fig.subplots_adjust(left=0.2, right=0.9, bottom=0.2, top=0.9)

    plt.xlabel('PC', fontsize=20, fontweight='bold')
    plt.ylabel('Variance', fontsize=20, fontweight='bold')
    plt.savefig(os.getcwd() + '/scree.png')
    plt.show()
    print("Proportion of Variance Explained : ", np.round(pca.explained_variance_ratio_, 4))

    out_sum = np.cumsum(np.round(pca.explained_variance_ratio_, 2))
    print("Cumulative Prop. Variance Explained: ", out_sum)

def elbow_plot(PCA_components):

  xloc = plticker.MultipleLocator(base=1.0) # this locator puts ticks at regular intervals
  yloc = plticker.MultipleLocator(base=100000) # this locator puts ticks at regular intervals

  wcss = []

  for i in range(1,11):
    kmeans = KMeans(n_clusters=i, init ='k-means++', max_iter=300,  n_init=10, random_state=0 )
    kmeans.fit(PCA_components)
    wcss.append(kmeans.inertia_)

  fig, ax = plt.subplots(figsize=(5, 5))
  ax.plot(range(1,11),wcss, color= 'darkcyan', linewidth = 2.0)
  # ax.xaxis.set_major_locator(xloc)
  # ax.yaxis.set_major_locator(yloc)
  # ax.set_ylim(0,(round(max(wcss), -5))+1)
  print(wcss)
  ax.set_ylim(0, round(max(wcss), -1) + 10)


  ax.set_xlabel('k', fontsize=20, fontweight='bold')
  ax.set_ylabel('WCSS', fontsize=20, fontweight='bold')

  ax.axes.ticklabel_format(axis='y', style='sci', scilimits=(0,0), useMathText=True)
  ax.spines['top'].set_lw(2)
  ax.spines['bottom'].set_lw(2)
  ax.spines['left'].set_lw(2)
  ax.spines['right'].set_lw(2)

  ax.tick_params(axis='both', which='major', labelsize=16)

  fig.subplots_adjust(left=0.2, right=0.9, bottom=0.2, top=0.9)

  plt.savefig(os.getcwd() + '/elbow.png')
  plt.show()

def counter(cluster):
    unique, counts = np.unique(cluster, return_counts=True)
    label_index = dict(zip(unique[1:4], counts[1:4]))
    return label_index

def plotter(label_dict, class_names):

    plt.bar(range(len(label_dict)), list(label_dict.values()), align='center', width=0.8, edgecolor='black', linewidth=1, color='darkgreen')
    a = []
    for i in [*label_dict]: a.append(class_names[i])
    plt.xticks(range(len(label_dict)), list(a), rotation=0, rotation_mode='anchor')
    plt.yticks()
    plt.xlabel('Sub-stage')
    plt.ylabel('Count')

def plot_counts(label_count):

  class_names = {1: '10.1', 2: '10.2', 3: '10.3'}
  #Bar graph with the number of items of different categories clustered in it
  plt.figure()
  plt.subplots_adjust(wspace = 0.8)
  mpl.rcParams['axes.linewidth'] = 2
  for i in range (1,3):
      plt.subplot(2, 2, i)
      plotter(label_count[i-1], class_names)
      plt.title("Cluster " + str(i))
  plt.savefig(os.getcwd() + '/counts.png')

def plot_scatter(k_means_labels, kmeans, PCA_components):
  plt.clf()
  plt.figure()

  ax = sns.scatterplot(x=PCA_components[0], y=PCA_components[1], hue=k_means_labels, palette = ['orange', 'blue'])
  ax = sns.scatterplot(x=kmeans.cluster_centers_[:, 0], y=kmeans.cluster_centers_[:, 1], color= 'r', s = 100)

  ax.set_ylabel('PC 2', fontsize = 14, fontweight='bold')
  ax.set_xlabel('PC 1', fontsize = 14, fontweight='bold')

  ax.spines['top'].set_lw(2)
  ax.spines['bottom'].set_lw(2)
  ax.spines['left'].set_lw(2)
  ax.spines['right'].set_lw(2)

  # Define a dictionary of cluster labels to colors
  cluster_colors = {1: 'orange', 2: 'blue'}

  # Create a custom legend with the desired labels and colors
  handles = [plt.plot([], [], marker="o", ms=10, ls="", mec=None, color=color,
                      label=label)[0] for label, color in cluster_colors.items()]
  ax.legend(handles=handles, labels=cluster_colors.keys(), title="Cluster", loc="upper right")

  # for handle in lgnd.legendHandles:
  #     handle.set_sizes([1.0])

  plt.tick_params(axis='both',which='both', bottom=False, top = False, labelbottom=False, labelleft=False)
  plt.savefig(os.getcwd() + '/scatter.png')

def save_opt_hyperparams(path, exp_name, best_hps):
    today = dt.today()
    date = today.strftime("%b-%d-%Y")

    os.makedirs(path + '\{}_{}'.format(exp_name, date), exist_ok=True)
    split_save_path = (path + '\{}_{}/opt_best_hps'.format(exp_name, date))
    to_save = best_hps
    with open(split_save_path, 'wb') as pickle_file:
        pkl.dump(to_save, pickle_file)

def load_opt_hyperparams(path):
    with open(path, 'rb') as pickle_file:
        best_hps = pkl.load(pickle_file)
    return best_hps

def create_data(path, duplicate_channels, limb=False, equalize=True):
    # os.chdir(path)
    data = []
    if limb:
        class_dirs = [path + '/control', path + '/treated',]
    else:
        class_dirs = [path + '/10_a', path + '/10_b',]

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


def load_test_set(split_save_path):
    with open(split_save_path, 'rb') as pickle_file:
        split_dict = pkl.load(pickle_file)

    return split_dict


import tensorflow as tf  # version 2.5
from tensorflow.keras import layers
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
from datetime import date as dt


def visualise_aug(X_train_aug, y_train_aug):
    for i in range(0, 36):
        sub = plt.subplot((6), 6, i + 1)
        sub.imshow(X_train_aug[0][i])
        sub.set_title('{}'.format(y_train_aug[0][i]))
    plt.show()


def reshape_and_normalize_TC(X, Y, nb_classes):
    shuffle(X)
    shuffle(Y)
    X_train, x_test, Y_train, y_test = train_test_split(X, Y, test_size=0.2)  # 70% training and 30% test
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

def reshape_and_normalize(X, Y, nb_classes):
    X = np.array(X).reshape(-1, 200, 200, 1)  # Array containing every pixel as an index (1D array - 40,000 long)
    X_train = np.array(X)
    Y_train = np.array(Y)
    print("The shape of X is " + str(X_train.shape))
    print("The shape of y is " + str(Y_train.shape))  # This is only used to check our clustering

    # Data Normalization

    print(X_train.min())  # Should be 0
    print(X_train.max())  # Should be 255

    # Conversion to float

    X_train = X_train.astype('float32')

    # Normalization

    X_train = X_train / 255.0

    print(X_train.min())  # Should be 0
    print(X_train.max())  # Should be 1.0

    # check that X has been correctly split into train and test sets
    print('X_train shape:', X_train.shape)
    print(X_train.shape[0], 'train samples')

    # convert class vectors to binary class matrices with one-hot encoding

    Y_train = to_categorical(Y_train, nb_classes)
    return X_train, Y_train

def fit_k_means(PCA_components, Y_train, number_of_clusters):

  kmeans = KMeans(n_clusters=number_of_clusters, init ='k-means++', max_iter=300, n_init=10,random_state=0)
  kmeans.fit(PCA_components)
  k_means_labels = kmeans.labels_ #List of labels of each dataset

  unique_labels = len(np.unique(k_means_labels))

  #2D matrix  for an array of indexes of the given label
  cluster_index= [[] for i in range(unique_labels)]
  for i, label in enumerate(k_means_labels,0):
      for n in range(unique_labels):

          if label == n:
              cluster_index[n].append(i)
          else:
              continue
  Y_clust = [[] for i in range(unique_labels)]
  for n in range(unique_labels):

      Y_clust[n] = Y_train[cluster_index[n]] #Y_clust[0] contains array of "correct" category from y_train for the cluster_index[0]
      assert(len(Y_clust[n]) == len(cluster_index[n])) #dimension confirmation


  for i in range(0, len(Y_clust)):
    for j in range(0, len(Y_clust[i])):
      if Y_clust[i][j][0] != 0.0:
        Y_clust[i][j][0] = 1.0
      elif Y_clust[i][j][1] != 0.0:
        Y_clust[i][j][1] = 2.0
      elif Y_clust[i][j][2] != 0.0:
        Y_clust[i][j][2] = 3.0
  return kmeans, k_means_labels, Y_clust, unique_labels


def fit_PCA(X_train, n_components):
    X = X_train.reshape(-1, X_train.shape[1] * X_train.shape[2])
    scaler = StandardScaler()

    pca = PCA(n_components)
    pca_fit = pca.fit(X)  # fit the data according to our PCA instance

    print("Number of components before PCA  = " + str(X.shape[1]))
    print("Number of components after PCA 2 = " + str(pca.n_components_))

    # dimension reduced from 40000 to 2
    scores_pca = pca.transform(X)
    return pca, pca_fit, scores_pca


def train_traditional_cf_model(train, val, train_label, val_label, name, results, hyperparams, i, model=None, limb = False, lr=0.00001, lmbd=0.0001):
    param_grid = {'C': [0.1, 1, 10, 100], 'gamma': [0.0001, 0.001, 0.1, 1], 'kernel': ['rbf', 'poly']}  # params for SVM
    print("type of train {}".format(type(train)))
    print("type of val {}".format(type(val)))
    print("len of train {}".format(len(train)))
    print("len of val {}".format(len(val)))
    # print("len of train_i {}".format(len(train[0])))
    # print("len of val_i {}".format(len(val[0])))

    np_train = np.array(train) / 255
    np_val = np.array(val) / 255

    reshaped_train = np_train
    reshaped_val = np_val

    reshaped_train = np_train.reshape(-1, 200, 200, 1)
    reshaped_val = np_val.reshape(-1, 200, 200, 1)

    reshaped_train.astype('float32')
    reshaped_val.astype('float32')

    # train_label = to_categorical(train_label)
    # val_label = to_categorical(val_label, 2)

    # reshaped_train = reshaped_train.reshape(-1, 1 * 200 * 200)
    # reshaped_val = reshaped_val.reshape(-1, 1 * 200 * 200)

    print('reshaped_train shape:', np.shape(reshaped_train))
    print('reshaped_val shape:', np.shape(reshaped_val))
    print('train_label shape:', np.shape(train_label))
    print('val_label shape:', np.shape(val_label))

    #
    # svc=svm.SVC(probability=True, verbose=1)
    # model=GridSearchCV(svc,param_grid)

    model = KNeighborsClassifier(n_neighbors=3)
    model.fit(reshaped_train,
                        train_label)
    y_pred=model.predict(reshaped_val)
    valaccs.append(metrics.accuracy_score(val_label, y_pred))


def train_model(train, val, train_label, val_label, X_test, Y_test, name, results, hyperparams, i, model=None, limb = False, lr=0.00001, lmbd=0.0001):

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
            # if layer.name not in ['dense_3']:
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
    # results["test_accuracies"].append((round(max(acc) * 100, 1)))

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
    print('y_test shape', np.shape(Y_test))  # should be (number of test labels, 3)
    # model.evaluate(X_test, Y_test)

    # results["test_performance"].append(model.evaluate(X_test, Y_test))
    results["test_accuracies"].append(model.evaluate(X_test, Y_test)[1])
    results["test_losses"].append(model.evaluate(X_test, Y_test)[0])
    # K.clear_session()
    del model


    return results, hyperparams


def train_model_resnet50(train, val, train_label, val_label, X_test, Y_test, name, results, hyperparams, i, model=None,
                         pretrained=False):
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
                        validation_data=(reshaped_val, val_label), verbose = 2, callbacks=[earlyStop])

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

    # reshaped_train = np_train.reshape(-1, 200, 200, 1)
    # reshaped_val = np_val.reshape(-1, 200, 200, 1)

    # reshaped_train.astype('float32')
    # reshaped_val.astype('float32')
    print("train shape before preprocess {}".format(np_train.shape))
    reshaped_train = preprocess_inception(np_train)
    reshaped_val = preprocess_inception(np_val)

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
        inception_model = InceptionV3(include_top=False, weights='imagenet', input_tensor=None, input_shape=(200, 200, 3),
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
                        validation_data=(reshaped_val, val_label), verbose = 2, callbacks=[earlyStop])

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

def train_model_vgg16(train, val, train_label, val_label, X_test, Y_test, name, results, hyperparams, i, model=None,
                         pretrained=False):
    np_train = np.array(train) / 255
    np_val = np.array(val) / 255

    # reshaped_train = np_train.reshape(-1, 200, 200, 1)
    # reshaped_val = np_val.reshape(-1, 200, 200, 1)

    # reshaped_train.astype('float32')
    # reshaped_val.astype('float32')
    print("train shape before preprocess {}".format(np_train.shape))
    reshaped_train = preprocess_vgg(np_train)
    reshaped_val = preprocess_vgg(np_val)

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
        print('building new vgg16 model')
        inception_model = VGG16(include_top=False, weights='imagenet', input_tensor=None, input_shape=(200, 200, 3),
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
                        validation_data=(reshaped_val, val_label), verbose = 2, callbacks=[earlyStop])

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


def aug_data_2(X_train, y_train, X_val, y_val, X_val_bool):
    X_train_aug = []
    X_val_aug = []
    y_train_aug = []
    y_val_aug = []
    X_train = np.array(X_train)
    X_val = np.array(X_val)
    print("augmenting data")
    for i in range(0, len(X_train)):

        for angle in np.arange(0, 360, 10):
            # seq = iaa.Sequential([iaa.Cutout(nb_iterations=(1, 3)), iaa.GaussianBlur(sigma = (0.0, 3.0)), iaa.ShearX((-20, 20))], random_order = True) #Random order but ALL
            # seq = iaa.SomeOf(1, [iaa.Cutout(nb_iterations=(1, 3)), iaa.GaussianBlur(sigma=(0.0, 3.0)),
            #                      iaa.ShearX((-20, 20))])  # Just one
            #
            # feature = seq(image=X_train[i])

            rotate = iaa.geometric.Affine(rotate=angle)
            feature = rotate(image=X_train[i])

            X_train_aug.append(feature)
            y_train_aug.append(y_train[i])
    if X_val_bool:
        for i in range(0, len(X_val)):

            for angle in np.arange(0, 360, 10):
                # seq = iaa.Sequential([iaa.Cutout(nb_iterations=(1, 3)), iaa.GaussianBlur(sigma = (0.0, 3.0)), iaa.ShearX((-20, 20))], random_order = True)
                seq = iaa.SomeOf(1, [iaa.Cutout(nb_iterations=(1, 3)), iaa.GaussianBlur(sigma=(0.0, 3.0)),
                                     iaa.ShearX((-20, 20))])  # Just one
                feature = seq(image=X_train[i])

                rotate = iaa.geometric.Affine(rotate=angle)
                feature = rotate(image=feature)

                X_val_aug.append(feature)
                y_val_aug.append(y_val[i])

        return X_train_aug, y_train_aug, X_val_aug, y_val_aug
    else:
        return X_train_aug, y_train_aug

def augment_data(X_train, y_train, X_val, y_val, baseline=False, cutout=False, shear=False, gblur=False, crop=False,
                 randcomb=False, mobius=False, allcomb_sparse=False, allcomb_full=False, resnet=False, inception=False, limb=False):
    print("resnet is "+ str(resnet))
    print("inception is" + str(inception))
    X_val_aug = []
    X_train_aug = []
    y_train_aug = []
    y_val_aug = []
    X_train = np.array(X_train)
    X_val = np.array(X_val)

    # Mode can be chosen from 'reflect’, ‘constant’, ‘nearest’, ‘mirror’, ‘wrap’
    mode = 'constant'

    # if user_defined == False, then it is random (recommended)
    user_defined = False
    start_points = 32, 16, 16, 32, 32, 48
    end_points = 16, 32, 32, 48, 48, 32

    for i in range(0, len(X_train)):  # for every 149 long list of images 0 -> 10

        rotated_features = []  # empty temp list of rotated images
        rotated_labels = []  # empty temp list of labels
        X_train[i] = np.array(X_train[i])  # convert list of images to np.array - seems to be necessary for imgaug

        for j in range(0, len(X_train[
                                  i])):  # for every image in 149 long list of images 0 -> 149 #uncomment for full length (very slow)

            # for j in range(0, 3): #todo try putting this inside the rotation loop so that each rotated image has a different mobius transform
            if mobius:
                # M must be >1
                # The smaller M is, the more "normal" the output looks
                M = np.linspace(1.1,1.2, 20)
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
                ## TODO fix baseline
                if cutout:
                    cutout = iaa.Cutout(nb_iterations=(1, 3),
                                        size=0.2)  # TODO try different number of iterations and various alternative args from imagaug docs
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
                if allcomb_sparse:
                    seq = iaa.Sequential([iaa.Cutout(nb_iterations=(1, 3)), iaa.GaussianBlur(sigma=(0.0, 3.0)),
                                         iaa.ShearX((-20, 20))], random_order=True)
                    feature = seq(image=X_train[i][j])

                if allcomb_full:

                    cutout = iaa.Cutout(nb_iterations=(1, 3),
                                        size=0.2)  # TODO try different number of iterations and various alternative args from imagaug docs
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
                if not resnet or inception:

                    feature = feature.convert("L")
                feature = np.array(feature)
                rotated_features.append(feature)  # append to 36 rotated images to temp
                rotated_labels.append(y_train[i][j])

        print("number of aug {}".format(len(rotated_labels)))  # todo assertion
        print("number of train pre aug {}".format(len(X_train[0])))  # todo assertion

        X_train_aug.append(
            rotated_features)  # when finished all j append 149 long list of 36 long list of rotated imgs, proceed to next i
        y_train_aug.append(rotated_labels)



        print("number of train post aug {}".format(len(X_train_aug[0])))  # todo assertion
        print("limb status {}".format(limb))
        # print("allcomb status {}".format(allcomb_full))
        if mobius:

            assert len(X_train_aug[0]) == (len(X_train[0]) * len(
                np.arange(0, 360,
                          10)) + len(X_train[0])), "X_train_aug is not equal to X_train multiplied by the number of transformations"

        if allcomb_full:
            if limb:
                print("here1")
                assert len(X_train_aug[0]) == (len(X_train[0]) * len(
                    np.arange(0, 360,
                              10))) * 5, "X_train_aug is not equal to X_train multiplied by the number of transformations"
            else:
                print("here2")
                assert len(X_train_aug[0]) == (len(X_train[0]) * len(
                    np.arange(0, 360,
                              10)) * 4), "X_train_aug is not equal to X_train multiplied by the number of transformations"
        if limb and (not allcomb_full):
            assert len(X_train_aug[0]) == (len(X_train[0]) * len(
                np.arange(0, 360,
                          10)))*2, "X_train_aug is not equal to X_train multiplied by the number of transformations"
        else:
            print("here3")
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
                M = np.linspace(1.1,1.2, 20)
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
                                        size=0.2)  # TODO try different number of iterations and various alternative args from imagaug docs
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
                                        size=0.2)  # TODO try different number of iterations and various alternative args from imagaug docs
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

                if allcomb_sparse:
                    seq = iaa.Sequential([iaa.Cutout(nb_iterations=(1, 3)), iaa.GaussianBlur(sigma=(0.0, 3.0)),
                                         iaa.ShearX((-20, 20))], random_order=True)
                    feature = seq(image=X_val[i][j])

                    rotated_features.append(feature)
                    rotated_labels.append(y_val[i][j])

                if limb:

                    flip = iaa.Fliplr(0.5)
                    flip_feature = flip(image=X_val[i][j])

                    rotated_features.append(flip_feature)
                    rotated_labels.append(y_val[i][j])

                rotate = iaa.geometric.Affine(rotate=angle)  # set up augmenter
                feature = rotate(image=X_val[i][j])  # rotate each image 36 times
                feature = Image.fromarray(feature)

                if not resnet or inception:
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
            print("X_val_aug len is {}".format(len(X_val_aug[0])))
            assert len(X_val_aug[0]) == (len(X_val[0]) * len(
                np.arange(0, 360,
                          10)))*2, "X_val_aug is not equal to X_val multiplied by the number of transformations"

        if allcomb_full:
            if limb:
                print("here1")
                assert len(X_val_aug[0]) == (len(X_val[0]) * len(
                    np.arange(0, 360,
                              10))) * 5, "X_train_aug is not equal to X_train multiplied by the number of transformations"
            else:
                print("here2")
                assert len(X_val_aug[0]) == (len(X_val[0]) * len(
                    np.arange(0, 360,
                              10)) * 4), "X_train_aug is not equal to X_train multiplied by the number of transformations"
        if limb and (not allcomb_full):
            assert len(X_val_aug[0]) == (len(X_val[0]) * len(
                np.arange(0, 360,
                          10)))*2, "X_train_aug is not equal to X_train multiplied by the number of transformations"
        else:
            print("X_val_aug len is {}".format(len(X_val_aug[0])))
            assert len(X_val_aug[0]) == (len(X_val[0]) * len(
                np.arange(0, 360,
                          10))), "X_val_aug is not equal to X_val multiplied by the number of transformations"

    print(len(X_train))
    print(len(X_train[i]))
    print(len(X_train[i][j]))

    print(len(X_train_aug))
    print(len(X_train_aug[i]))
    print(len(X_train_aug[i][j]))

    # assert len(X_train_aug[0]) == len(X_train[0]) * len(
    #     np.arange(0, 360, 10)), "X_train_aug is not equal to X_train multiplied by the number of transformations"

    return X_train_aug, y_train_aug, X_val_aug, y_val_aug


def augment_data_hd_cutout(X_train, y_train, X_val, y_val, cutout=False, randcomb=False):
    if cutout:
        print('augmentation is cutout')
    if randcomb:
        print('augmentation is randcomb')

    X_train_aug = []
    X_val_aug = []
    y_train_aug = []
    y_val_aug = []
    X_train = np.array(X_train)
    X_val = np.array(X_val)

    for i in range(0, len(X_train)):

        for angle in np.arange(0, 360, 10):

            if cutout:
                cutout = iaa.Cutout(nb_iterations=(1, 3),
                                    size=0.2)  # TODO try different number of iterations and various alternative args from imagaug docs
                feature = cutout(image=X_train[i])

            if randcomb:
                seq = iaa.SomeOf(1, [iaa.Cutout(nb_iterations=(1, 3)), iaa.GaussianBlur(sigma=(0.0, 3.0)),
                                     iaa.ShearX((-20, 20))])  # Just one
                feature = seq(image=X_train[i])

            rotate = iaa.geometric.Affine(rotate=angle)
            feature = rotate(image=X_train[i])

            X_train_aug.append(feature)
            y_train_aug.append(y_train[i])

    for i in range(0, len(X_val)):

        for angle in np.arange(0, 360, 10):

            if cutout:
                cutout = iaa.Cutout(nb_iterations=(1, 3),
                                    size=0.2)  # TODO try different number of iterations and various alternative args from imagaug docs
                feature = cutout(image=X_val[i])

            if randcomb:
                seq = iaa.SomeOf(1, [iaa.Cutout(nb_iterations=(1, 3)), iaa.GaussianBlur(sigma=(0.0, 3.0)),
                                     iaa.ShearX((-20, 20))])  # Just one
                feature = seq(image=X_val[i])

            rotate = iaa.geometric.Affine(rotate=angle)
            feature = rotate(image=X_val[i])

            X_val_aug.append(feature)
            y_val_aug.append(y_val[i])

    return X_train_aug, X_val_aug, y_train_aug, y_val_aug
#

# def augment_data_holdout(X_train, y_train, X_val, y_val, baseline=False, cutout=False, shear=False, gblur=False,
#                          crop=False, randcomb=False, mobius=False):
#     X_val_aug = []
#     X_train_aug = []
#     y_train_aug = []
#     y_val_aug = []
#     X_train = np.array(X_train)
#     X_val = np.array(X_val)
#
#     # Mode can be chosen from 'reflect’, ‘constant’, ‘nearest’, ‘mirror’, ‘wrap’
#     mode = 'constant'
#
#     # if user_defined == False, then it is random (recommended)
#     user_defined = False
#     start_points = 32, 16, 16, 32, 32, 48
#     end_points = 16, 32, 32, 48, 48, 32
#     print('X_train_len_{}'.format(len(X_train)))
#     print('X_val_len_{}'.format(len(X_val)))
#     print('y_train_len_{}'.format(len(y_train)))
#     print('y_val_len_{}'.format(len(y_val)))
#
#     for i in range(0, len(X_train)):  # for every 149 long list of images 0 -> 10
#
#         rotated_features = []  # empty temp list of rotated images
#         rotated_labels = []  # empty temp list of labels
#         X_train = np.array(X_train)  # convert list of images to np.array - seems to be necessary for imgaug
#
#         if mobius:
#             # M must be >1
#             # The smaller M is, the more "normal" the output looks
#             M = np.linspace(5, 6, 20)
#             M = np.random.choice(M)
#             print(M)
#
#             img = X_train[i]
#             feature, uninterpolated_image = mobius_fast_interpolation('example', True, img,
#                                                                       M,
#                                                                       mode=mode,
#                                                                       output_height=200,
#                                                                       output_width=200,
#                                                                       user_defined=user_defined,
#                                                                       start_points=start_points,
#                                                                       end_points=end_points)
#             feature = np.array(feature)
#
#         for angle in np.arange(0, 360, 10):
#             # TODO fix baseline
#             if cutout:
#                 cutout = iaa.Cutout(nb_iterations=(1, 3),
#                                     size=0.2)  # TODO try different number of iterations and various alternative args from imagaug docs
#                 feature = cutout(image=X_train[i])
#             if shear:
#                 shear = iaa.ShearX((-20, 20))
#                 feature = shear(image=X_train[i])
#             if gblur:
#                 gblur = iaa.GaussianBlur(sigma=(0.0, 5.0))  # sigma = 5 was originally used - not random
#                 feature = gblur(image=X_train[i])
#             if crop:
#                 crop = iaa.Crop(percent=(0.0, 0.3))  # originally every img was cropped 30 times
#                 feature = crop(image=X_train[i])
#             if randcomb:
#                 seq = iaa.SomeOf(1, [iaa.Cutout(nb_iterations=(1, 3)), iaa.GaussianBlur(sigma=(0.0, 3.0)),
#                                      iaa.ShearX((-20, 20))])  # Just one
#                 feature = seq(image=X_train[i])
#             else:
#                 rotate = iaa.geometric.Affine(rotate=angle)  # set up augmenter
#                 feature = rotate(image=X_train[i])  # rotate each image 36 times
#                 rotated_features.append(feature)
#                 rotated_labels.append(y_train[i])
#
#         X_train_aug.append(rotated_features)  # append to 36 rotated images to temp
#         X_val_aug.append(rotated_labels)
#
#         print('X_train_aug len is {}'.format(len(X_train_aug[i])))
#         print('X_train_len is {}'.format(len(X_train)))
#         assert len(X_train_aug) == len(X_train) * len(
#             np.arange(0, 360, 10)), "X_train_aug is not equal to X_train multiplied by the number of transformations"
#
#     for i in range(0, len(X_val)):  # for every 149 long list of images 0 -> 10 #TODO implement mobius for val
#
#         rotated_features = []  # empty temp list of rotated images
#         rotated_labels = []  # empty temp list of labels
#         X_val = np.array(X_val)  # convert list of images to np.array - seems to be necessary for imgaug
#
#         for angle in np.arange(0, 360, 10):
#
#             if cutout:
#                 cutout = iaa.Cutout(nb_iterations=(1, 3),
#                                     size=0.2)  # TODO try different number of iterations and various alternative args from imagaug docs
#                 feature = cutout(image=X_val[i])
#             if shear:
#                 shear = iaa.ShearX((-20, 20))
#                 feature = shear(image=X_val[i])
#             if gblur:
#                 gblur = iaa.GaussianBlur(sigma=(0.0, 5.0))  # sigma = 5 was originally used - not random
#                 feature = gblur(image=X_val[i])
#             if crop:
#                 crop = iaa.Crop(percent=(0.0, 0.3))  # originally every img was cropped 30 times
#                 feature = crop(image=X_val[i])
#             if randcomb:
#                 seq = iaa.SomeOf(1, [iaa.Cutout(nb_iterations=(1, 3)), iaa.GaussianBlur(sigma=(0.0, 3.0)),
#                                      iaa.ShearX((-20, 20))])  # Just one
#                 feature = seq(image=X_val[i])
#             if baseline:
#                 rotate = iaa.geometric.Affine(rotate=angle)  # set up augmenter
#                 feature = rotate(image=feature)  # rotate each image 36 times
#                 rotated_features.append(feature)  # append to 36 rotated images to temp
#                 rotated_labels.append(y_val[i])
#
#         X_val_aug.append(
#             rotated_features)  # when finished all j append 149 long list of 36 long list of rotated imgs, proceed to next i
#         y_val_aug.append(rotated_labels)
#
#     print(len(X_train))
#     print(len(X_train[i]))
#
#     print(len(X_train_aug))
#     print(len(X_train_aug[i]))
#
#     # assert len(X_train_aug[0]) == len(X_train[0]) * len(
#     #     np.arange(0, 360, 10)), "X_train_aug is not equal to X_train multiplied by the number of transformations"
#
#     return X_train_aug, y_train_aug, X_val_aug, y_val_aug


import numpy as np
from PIL import Image
import random
from scipy.ndimage import geometric_transform, map_coordinates
from numpy import *
from random import random


# TODO decide whether I need random seeds

def shift_func(coords, a, b, c, d):
    """ Define the mobius transformation, though backwards """
    # turn the first two coordinates into an imaginary number
    z = coords[0] + 1j * coords[1]
    w = (d * z - b) / (-c * z + a)  # the inverse mobius transform
    # take the color along for the ride
    return real(w), imag(w), coords[2]


def mobius_fast_interpolation(name, save, image, M, mode, rgb, output_height=None, output_width=None,
                              user_defined=False, start_points=None, end_points=None):
    image = np.array(image)
    original_image = image
    height = image.shape[0]
    width = image.shape[1]

    # User can pick output size
    if output_height == None:
        output_height = height
    if output_width == None:
        output_width = width
    if user_defined == True:
        # Method one
        # You pick starting and ending point
        a, b, c, d, original_points, new_points = getabcd_1fix(height, width, start_points, end_points)
    else:
        # Method two
        # Randomly generated starting the ending point
        a, b, c, d, original_points, new_points = madmissable_abcd(M, height, width)
    e = [complex(0, 0)] * height * width
    z = np.array(e).reshape(height, width)
    for i in range(0, height):
        for j in range(0, width):
            z[i, j] = complex(i, j)
    i = np.array(list(range(0, height)) * width).reshape(width, height).T
    j = np.array(list(range(0, width)) * height).reshape(height, width)

    r = ones((output_height, output_width, 3), dtype=uint8) * 255 * 0
    w = (a * z + b) / (c * z + d)
    first = real(w) * 1
    second = imag(w) * 1
    first = first.astype(int)
    second = second.astype(int)

    f1 = first >= 0
    f2 = first < output_height
    f = f1 & f2
    s1 = second >= 0
    s2 = second < output_width
    s = s1 & s2

    combined = s & f

    r[first[combined], second[combined], :] = image[i[combined], j[combined], :]

    r_interpolated = r.copy()
    u = [True] * output_height * output_width
    canvas = np.array(u).reshape(output_height, output_width)
    canvas[first[combined], second[combined]] = False
    converted_empty_index = np.where(canvas == True)
    converted_first = converted_empty_index[0]
    converted_second = converted_empty_index[1]

    new = converted_first.astype(complex)
    new.imag = converted_second

    ori = (d * new - b) / (-c * new + a)

    p = np.hstack([ori.real, ori.real, ori.real])
    k = np.hstack([ori.imag, ori.imag, ori.imag])
    zero = np.zeros_like(ori.real)
    one = np.ones_like(ori.real)
    two = np.ones_like(ori.real) * 2
    third = np.hstack([zero, one, two])
    number_of_interpolated_point = len(one)
    e = number_of_interpolated_point
    interpolated_value_unfinished = map_coordinates(image, [p, k, third], order=1, mode=mode, cval=0)
    t = interpolated_value_unfinished

    interpolated_value = np.stack([t[0:e], t[e:2 * e], t[2 * e:]]).T

    r_interpolated[converted_first, converted_second, :] = interpolated_value

    new_image = Image.fromarray(r_interpolated)
    # new_image_c1, new_image_c2, new_image_c3 = new_image.split()
    uninterpolated_image = Image.fromarray(r)

    if not rgb:
        new_image = new_image.convert("L")
        return new_image, uninterpolated_image
    new_image_c1 = new_image_c1.convert("L")
    return new_image, uninterpolated_image

def aug_mobius(X_train, y_train, X_val, y_val, M=5, mode='constant', user_defined=False, rgb=False):

    X_val_aug = []
    X_train_aug = []
    y_train_aug = []
    y_val_aug = []
    X_train = np.array(X_train)
    X_val = np.array(X_val)

    start_points = 32, 16, 16, 32, 32, 48
    end_points = 16, 32, 32, 48, 48, 32
    for i in range(0, len(X_train)):  # for every 149 long list of images 0 -> 10

        rotated_features = []  # empty temp list of rotated images
        rotated_labels = []  # empty temp list of labels
        X_train[i] = np.array(X_train[i])  # convert list of images to np.array - seems to be necessary for imgaug

        for j in range(0, len(X_train[i])):  # for every image in 149 long list of images 0 -> 149
                img = X_train[i][j]
            # do the mobius transforms
                feature, uninterpolated_feature = mobius_fast_interpolation('example', True, img,
                                                                            M,
                                                                            mode=mode, rgb=rgb,
                                                                            output_height=200,
                                                                            output_width=200,
                                                                            user_defined=user_defined,
                                                                            start_points=start_points,
                                                                            end_points=end_points)
                feature = np.array(feature)
                rotated_features.append(feature)
                rotated_labels.append(y_train[i][j])
                for angle in np.arange(0,360,10):
                    rotate = iaa.geometric.Affine(rotate=angle)  # set up augmenter
                    feature = rotate(image=X_train[i][j])  # rotate each image 36 times
                    feature = Image.fromarray(feature)
                    feature = feature.convert("L")
                    feature = np.array(feature)
                    rotated_features.append(feature)  # append to 36 rotated images to temp
                    rotated_labels.append(y_train[i][j])




        X_train_aug.append(rotated_features)
        y_train_aug.append(rotated_labels)

    for i in range(0, len(X_val)):  # for every 149 long list of images 0 -> 10

        rotated_features_val = []  # empty temp list of rotated images
        rotated_labels_val = []  # empty temp list of labels
        X_val[i] = np.array(X_val[i])  # convert list of images to np.array - seems to be necessary for imgaug

        for j in range(0, len(X_val[i])):  # for every image in 149 long list of images 0 -> 149

            #do the mobius transforms
            img = X_val[i][j]
            feature, uninterpolated_feature = mobius_fast_interpolation('example', True, img,
                                                                      M,
                                                                      mode = mode, rgb = rgb,
                                                                      output_height=200,
                                                                      output_width=200,
                                                                      user_defined=user_defined,
                                                                      start_points = start_points,
                                                                      end_points = end_points)
            feature = np.array(feature)
            rotated_features_val.append(feature)
            rotated_labels_val.append(y_val[i][j])

            for angle in np.arange(0, 360, 10):
                rotate = iaa.geometric.Affine(rotate=angle)  # set up augmenter
                feature = rotate(image=X_val[i][j])  # rotate each image 36 times
                feature = Image.fromarray(feature)
                feature = feature.convert("L")
                feature = np.array(feature)
                rotated_features_val.append(feature)  # append to 36 rotated images to temp
                rotated_labels_val.append(y_val[i][j])

        X_val_aug.append(rotated_features_val)
        y_val_aug.append(rotated_labels_val)
    return X_train_aug, y_train_aug, X_val_aug, y_val_aug

#
# def mobius_fast_interpolation(name, save, image, M, mode, output_height=None, output_width=None, user_defined=False,
#                               start_points=None, end_points=None):
#     image = np.array(image)
#     original_image = image
#     height = image.shape[0]
#     width = image.shape[1]
#
#     # User can pick output size
#     if output_height == None:
#         output_height = height
#     if output_width == None:
#         output_width = width
#     if user_defined == True:
#         # Method one
#         # You pick starting and ending point
#         a, b, c, d, original_points, new_points = getabcd_1fix(height, width, start_points, end_points)
#     else:
#         # Method two
#         # Randomly generated starting the ending point
#         a, b, c, d, original_points, new_points = madmissable_abcd(M, height, width)
#     e = [complex(0, 0)] * height * width
#     z = np.array(e).reshape(height, width)
#     for i in range(0, height):
#         for j in range(0, width):
#             z[i, j] = complex(i, j)
#     i = np.array(list(range(0, height)) * width).reshape(width, height).T
#     j = np.array(list(range(0, width)) * height).reshape(height, width)
#
#     r = ones((output_height, output_width, 3), dtype=uint8) * 255 * 0
#     w = (a * z + b) / (c * z + d)
#     first = real(w) * 1
#     second = imag(w) * 1
#     first = first.astype(int)
#     second = second.astype(int)
#
#     f1 = first >= 0
#     f2 = first < output_height
#     f = f1 & f2
#     s1 = second >= 0
#     s2 = second < output_width
#     s = s1 & s2
#
#     combined = s & f
#
#     r[first[combined], second[combined], :] = image[i[combined], j[combined], :]
#
#     r_interpolated = r.copy()
#     u = [True] * output_height * output_width
#     canvas = np.array(u).reshape(output_height, output_width)
#     canvas[first[combined], second[combined]] = False
#     converted_empty_index = np.where(canvas == True)
#     converted_first = converted_empty_index[0]
#     converted_second = converted_empty_index[1]
#
#     new = converted_first.astype(complex)
#     new.imag = converted_second
#
#     ori = (d * new - b) / (-c * new + a)
#
#     p = np.hstack([ori.real, ori.real, ori.real])
#     k = np.hstack([ori.imag, ori.imag, ori.imag])
#     zero = np.zeros_like(ori.real)
#     one = np.ones_like(ori.real)
#     two = np.ones_like(ori.real) * 2
#     third = np.hstack([zero, one, two])
#     number_of_interpolated_point = len(one)
#     e = number_of_interpolated_point
#     interpolated_value_unfinished = map_coordinates(image, [p, k, third], order=1, mode=mode, cval=0)
#     t = interpolated_value_unfinished
#
#     interpolated_value = np.stack([t[0:e], t[e:2 * e], t[2 * e:]]).T
#
#     r_interpolated[converted_first, converted_second, :] = interpolated_value
#
#     new_image = Image.fromarray(r_interpolated)
#     uninterpolated_image = Image.fromarray(r)
#     new_image = new_image.convert("L")
#     uninterpolated_image = uninterpolated_image.convert("L")
#     new_image_arr = np.array(new_image)
#     uninterpolated_image_arr = np.array(uninterpolated_image)
#
#     # fig = figure(figsize=(15, 10), dpi=300)
#     # subplot(1,3,1)
#     # title('Original', fontsize = 20)
#     # axis('off')
#
#     # #imshow(original_image)
#     # subplot(1,3,2)
#     # title('No interpolation. M = {}'.format(M), fontsize = 20)
#     # axis('off')
#     # imshow(r)
#     # subplot(1,3,3)
#     # # figure()
#     # title('With interpolation. M = {}'.format(M), fontsize = 20)
#     # axis('off')
#     # imshow(r_interpolated)
#     # if save:
#     #   fig.savefig('/content/drive/MyDrive/9. ML project/Figures/mobius_example_{}_{}.png'.format(name, M))
#
#     # return new_image, uninterpolated_image
#     return new_image_arr, uninterpolated_image_arr


def getabcd_1fix(height, width, start_points, end_points):
    # fixed start and end points

    start1_x, start1_y, start2_x, start2_y, start3_x, start3_y = start_points
    end1_x, end1_y, end2_x, end2_y, end3_x, end3_y = end_points
    zp = [complex(start1_x, start1_y), complex(start2_x, start2_y), complex(start3_x, start3_y)]
    wa = [complex(end1_x, end1_y), complex(end2_x, end2_y), complex(end3_x, end3_y)]

    # This is for plotting points on the output, not useful for calculation
    original_points = np.array([[start1_x, start1_y], [start2_x, start2_y], [start3_x, start3_y]], dtype=int)
    new_points = np.array([[end1_x, end1_y], [end2_x, end2_y], [end3_x, end3_y]], dtype=int)

    a = np.linalg.det([[zp[0] * wa[0], wa[0], 1],
                       [zp[1] * wa[1], wa[1], 1],
                       [zp[2] * wa[2], wa[2], 1]]);
    b = np.linalg.det([[zp[0] * wa[0], zp[0], wa[0]],
                       [zp[1] * wa[1], zp[1], wa[1]],
                       [zp[2] * wa[2], zp[2], wa[2]]]);

    c = np.linalg.det([[zp[0], wa[0], 1],
                       [zp[1], wa[1], 1],
                       [zp[2], wa[2], 1]]);

    d = np.linalg.det([[zp[0] * wa[0], zp[0], 1],
                       [zp[1] * wa[1], zp[1], 1],
                       [zp[2] * wa[2], zp[2], 1]]);

    return a, b, c, d, original_points, new_points


# Test if a, b, c, and d fit our criteria
def M_admissable(M, a, b, c, d):
    size = 32
    v1 = np.absolute(a) ** 2 / np.absolute(a * d - b * c)
    if not (v1 < M and v1 > 1 / M):
        return False

    v2 = np.absolute(a - size * c) ** 2 / (np.absolute(a * d - b * c))
    if not (v2 < M and v2 > 1 / M):
        return False

    v3 = np.absolute(complex(a, -size * c)) ** 2 / np.absolute(a * d - b * c)
    if not (v3 < M and v3 > 1 / M):
        return False

    v4 = np.absolute(complex(a - size * c, -size * c)) ** 2 / np.absolute(a * d - b * c)
    if not (v4 < M and v4 > 1 / M):
        return False

    v5 = np.absolute(complex(a - size / 2 * c, -size / 2 * c)) ** 2 / (np.absolute(a * d - b * c))
    if not (v5 < M and v5 > 1 / M):
        return False

    v6 = np.absolute(
        complex(size / 2 * d - b, size / 2 * d) / complex(a - size / 2 * c, -size / 2 * c) - complex(size / 2,
                                                                                                     size / 2))
    if not (v6 < size / 4):
        return False

    return True


def madmissable_abcd(M, height, width):
    test = False
    while test == False:
        # Zp are the start points (3 points)
        # Wa are the end points  (3 points)
        zp = [complex(height * random(), width * random()), complex(height * random(), width * random()),
              complex(height * random(), width * random())]
        wa = [complex(height * random(), width * random()), complex(height * random(), width * random()),
              complex(height * random(), width * random())]

        # This is for ploting points on the output, not useful for calculation
        original_points = np.array([[real(zp[0]), imag(zp[0])],
                                    [real(zp[1]), imag(zp[1])],
                                    [real(zp[2]), imag(zp[2])]], dtype=int)
        new_points = np.array([[real(wa[0]), imag(wa[0])],
                               [real(wa[1]), imag(wa[1])],
                               [real(wa[2]), imag(wa[2])]], dtype=int)

        # transformation parameters
        a = linalg.det([[zp[0] * wa[0], wa[0], 1],
                        [zp[1] * wa[1], wa[1], 1],
                        [zp[2] * wa[2], wa[2], 1]]);

        b = linalg.det([[zp[0] * wa[0], zp[0], wa[0]],
                        [zp[1] * wa[1], zp[1], wa[1]],
                        [zp[2] * wa[2], zp[2], wa[2]]]);

        c = linalg.det([[zp[0], wa[0], 1],
                        [zp[1], wa[1], 1],
                        [zp[2], wa[2], 1]]);

        d = linalg.det([[zp[0] * wa[0], zp[0], 1],
                        [zp[1] * wa[1], zp[1], 1],
                        [zp[2] * wa[2], zp[2], 1]]);
        test = M_admissable(M, a, b, c, d)

    return a, b, c, d, original_points, new_points