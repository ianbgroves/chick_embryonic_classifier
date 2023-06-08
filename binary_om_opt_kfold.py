import tensorflow as tf
import keras_tuner as kt
from tensorflow import keras
from tensorflow.keras import layers
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
import gc
from datetime import date as dt
from binary_utils import *
import numpy as np
import time
from sklearn.model_selection import train_test_split

print(f"TensorFlow Version: {tf.__version__}")
print(f"KerasTuner Version: {kt.__version__}")

today = dt.today()
date = today.strftime("%b-%d-%Y")
os.chdir('G:/My Drive/Python_projects/classifier/binary_classification')
exp_name, baseline, cutout, shear, gblur, crop, randcomb, mobius, allcomb_sparse, allcomb_full = read_args()
print('exp_name '+exp_name)
load = True
if load:
    split_dict = load_test_set("saved_test_sets/binary_baseline_3_Mar-14-2023/pkl_splits")
    print(split_dict.keys())
    X = split_dict['X']
    Y = split_dict['Y']
    X_test = split_dict['X_test']
    y_test = split_dict['y_test']

else:

    data = create_data(os.path.join(os.getcwd(), 'merged'), duplicate_channels=False)
    print('len of data is {}'.format(len(data)))
    data_list = []
    data_list.append(data[0:len(data)])

    X = []
    Y = []
    for i in data_list:
        for feature, label in i:
            X.append(feature)
            Y.append(label)

X_train, y_train, X_val, y_val = kfoldcv(X, Y, k=10)

X_train_aug, y_train_aug, X_val_aug, y_val_aug = augment_data(X_train, y_train, X_val, y_val, baseline, cutout, shear,
                                                              gblur, crop, randcomb, mobius, allcomb_sparse, allcomb_full)


def build_hp_opt_model(hp):
    """
    Hyperparameter optimisation function
    :param hp: Object containing which hyperparameters and the ranges over which to tune
    :return: model - keras model
    """
    hp_activation = hp.Choice('activation', values = ['relu', 'sigmoid'])
    hp_layer_drop = hp.Choice("dropout", values = [0.2, 0.3])
    hp_final_drop = hp.Choice("final_drop", values = [0.5, 0.6])
    hp_lr = hp.Choice("learning_rate", values = [0.001, 0.0001, 0.00001])
    hp_lmbd = hp.Choice("l2_lmbd", values = [0.001, 0.0001])

    # lmbd = hp.Float("l2_lmbd", 0.00001, 0.1, step=0.1, default=0.001)
    # lr = hp.Float("learning_rate", 0.00001, 0.001, step=0.1, default=0.001)
    # final_drop = hp.Float("final_dropout", 0.1, 0.5, step=0.1)
    # hp_optimizer=hp.Choice('optimizer', values=['adam', 'SGD'])
    # classification_units = hp.Int('classification_units', min_value=512, max_value=2048, step=128)
    num_layers = hp.Int('classification_layers', 2, 3)

    activation_mapping = {
        'relu': 'relu',
        'sigmoid': 'sigmoid'
    }

    layer_drop_mapping = {
        0.2: 0.2,
        0.3: 0.3
    }
    final_drop_mapping = {
        0.5 : 0.5,
        0.6 : 0.6
    }
    lmbd_mapping = {
        0.001: 0.001,
        0.0001: 0.0001
    }

    lr_mapping = {
        0.001: 0.001,
        0.0001: 0.0001,
        0.00001: 0.00001

    }

    activation = activation_mapping.get(hp_activation, None)
    layer_drop = layer_drop_mapping.get(hp_layer_drop, None)
    final_drop = final_drop_mapping.get(hp_final_drop, None)
    lmbd = lmbd_mapping.get(hp_lmbd, None)
    lr = lr_mapping.get(hp_lr, None)

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
        layers.Dense(1024, activation = activation)])

    for i in range(num_layers):
        model.add(layers.Dense(2048, activation=activation)),

    model.add(layers.Dropout(final_drop)),
    model.add(layers.Dense(2, activation='softmax'))

    #    if hp_optimizer == 'SGD':
    #        optimizer=SGD(learning_rate=lr)
    #    if hp_optimizer == 'adam':

    optimizer = Adam(learning_rate=lr)

    model.compile(optimizer=optimizer,
                  loss='binary_crossentropy',
                  metrics=['accuracy'])

    return model


def reshaper(X_train_aug, X_val_aug, y_train_aug, y_val_aug):
    print("len X_train {}".format(np.shape(X_train_aug)))
    print("len y_train {}".format(len(y_train_aug)))
    print("len X_val {}".format(len(X_val_aug)))
    print("len y_val {}".format(len(y_val_aug)))

    np_train = np.array(X_train_aug) / 255
    np_val = np.array(X_val_aug) / 255

    reshaped_train = np_train.reshape(-1, 200, 200, 1)
    reshaped_val = np_val.reshape(-1, 200, 200, 1)

    reshaped_train.astype('float32')
    reshaped_val.astype('float32')

    train_label = to_categorical(y_train_aug)
    val_label = to_categorical(y_val_aug, 2)

    print('reshaped_train shape:', np.shape(reshaped_train))
    print('reshaped_val shape:', np.shape(reshaped_val))
    print('train_label shape:', np.shape(train_label))
    print('val_label shape:', np.shape(val_label))

    return reshaped_train, reshaped_val, train_label, val_label


best_hps = []
results = {"accuracies": [], "losses": [], "val_accuracies": [],
           "val_losses": [], "test_performance": [], "test_accuracies": [], "test_losses": []}
EarlyStop = tf.keras.callbacks.EarlyStopping(monitor='val_loss', min_delta=0.01, patience=10, restore_best_weights=True)
for i in range(0, len(X_train_aug)):
    t0 = time.time()
    reshaped_train, reshaped_val, train_label, val_label = reshaper(X_train_aug[i], X_val_aug[i], y_train_aug[i],
                                                                    y_val_aug[i])

    tuner = kt.tuners.BayesianOptimization(
        build_hp_opt_model,
        objective='val_accuracy',
        max_trials=18,
        overwrite=True,
        directory='kt_dir',
        project_name='{}_{}'.format(exp_name, i))

    print("Searching for fold = {}".format(i))
    tuner.search_space_summary()
    tuner.search(reshaped_train, train_label, epochs=100, validation_data=(reshaped_val, val_label), callbacks=[EarlyStop], verbose=2)

    best_hps.append(tuner.get_best_hyperparameters()[0])
#   model = tuner.hypermodel.build(best_hps[0])
    model = tuner.get_best_models(num_models=1)[0]
    model.save(os.path.join("G:/My Drive/Python_projects/classifier/binary_classification/saved_models", exp_name,
                            '_{}_{}'.format(date, i)))

    X_test = np.array(X_test) / 255
    X_test = X_test.reshape(X_test.shape[0], 200, 200, 1)
    Y_test = to_categorical(y_test)

    print('X_test shape:', np.shape(X_test))  # should be (number of test images, 200, 200, 1)
    print('y_test shape', np.shape(Y_test))  # should be (number of test labels, 3)

    results["test_accuracies"].append(model.evaluate(X_test, Y_test)[1])
    results["test_losses"].append(model.evaluate(X_test, Y_test)[0])
    t1 = time.time()
    print("Fold took", int(t1 - t0), "sec to run")

    K.clear_session()
    tf.compat.v1.reset_default_graph()
    gc.collect()

save_opt_hyperparams(os.path.join(os.getcwd() + '/kt_dir'), exp_name, best_hps)
print('Results:', results, file=open('Results_{}_{}.txt'.format(exp_name, date), "w"))
