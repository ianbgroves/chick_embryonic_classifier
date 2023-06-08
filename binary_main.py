from datetime import date as dt
import os
import gc
import pickle as pkl  # module for serialization
from tensorflow.keras import backend as K

today = dt.today()
date = today.strftime("%b-%d-%Y")
from binary_utils import *

load = True
resnet = False
inception = True

os.chdir(r'G:/My Drive/Python_projects/classifier/binary_classification')
exp_name, baseline, cutout, shear, gblur, crop, randcomb, mobius, allcomb_sparse, allcomb_full = read_args()
print('exp_name '+exp_name)
if load:
    split_dict = load_test_set("saved_test_sets/binary_baseline_3_Mar-14-2023/pkl_splits")
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

    if mobius or resnet or inception:
        data = create_data(os.path.join(os.getcwd(), 'data_10a_b'), duplicate_channels=True)
    else:
        print(os.getcwd())
        data = create_data(os.path.join(os.getcwd(), 'data_10a_b'), duplicate_channels=False)

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
        "ratios of labels in the data set are {} {} {}".format(round(Y.count(0) / len(Y), 2), round(Y.count(1) / len(Y), 2),
                                                               round(Y.count(2) / len(Y), 2)))

    X, X_test, Y, y_test = train_test_split(X, Y, test_size=0.2, stratify=Y)
    print("ratios of labels in the test set are {} : {} : {}".format(round(y_test.count(0) / len(y_test), 2),
                                                                     round(y_test.count(1) / len(y_test), 2),
                                                                     round(y_test.count(2) / len(y_test), 2)))

if mobius or resnet or inception:
    print("converting to RGB")
    for i in range(0, len(X)):
        X[i] = Image.fromarray(X[i])
        X[i] = X[i].convert("RGB")
        X[i] = np.array(X[i])


    for i in range(0, len(X_test)):
        X_test[i] = Image.fromarray(X_test[i])
        X_test[i] = X_test[i].convert("RGB")
        X_test[i] = np.array(X_test[i])

if not load:
  split_dict = save_test_set('saved_test_sets', exp_name, X, X_test, Y, y_test)


# Kfold CV (k=10)
X_train, y_train, X_val, y_val = kfoldcv(X, Y, k=10)

X_train_aug, y_train_aug, X_val_aug, y_val_aug = augment_data(X_train, y_train, X_val, y_val, baseline, cutout, shear,
                                                              gblur, crop, randcomb, mobius, allcomb_sparse, allcomb_full, resnet, inception)
# X_train_aug = np.array(X_train_aug[i])
# print("shape of aug " + str(X_train_aug.shape))

results = {"accuracies": [], "losses": [], "val_accuracies": [],
           "val_losses": [], "test_performance": [], "test_accuracies": [], "test_losses": []}
hyperparams = {"configuration": [], "loss_func": [], "optimizer": [], "learning_rate": [], "lambda": []}

# if resnet:
#     for i in range(0, len(X_train_aug)):
#         K.clear_session()
#         tf.compat.v1.reset_default_graph()
#         gc.collect()
#
#         print("training_model_{}".format(i))
#         results = train_model_resnet50(X_train_aug[i], X_val_aug[i], y_train_aug[i], y_val_aug[i], X_test, y_test,
#                                        exp_name, results, hyperparams, i, model = None, pretrained= False)
#         K.clear_session()
#         tf.compat.v1.reset_default_graph()
#         gc.collect()

# if resnet:
#     for i in range(0, len(X_train_aug)):
#         K.clear_session()
#         tf.compat.v1.reset_default_graph()
#         gc.collect()
#
#         print("training_inception_model_{}".format(i))
#         results = train_model_inception(X_train_aug[i], X_val_aug[i], y_train_aug[i], y_val_aug[i], X_test, y_test,
#                                        exp_name, results, hyperparams, i, model = None, pretrained= False)
#         K.clear_session()
#         tf.compat.v1.reset_default_graph()
#         gc.collect()

if resnet:
    for i in range(0, len(X_train_aug)):
        K.clear_session()
        tf.compat.v1.reset_default_graph()
        gc.collect()

        print("training_vgg16_model_{}".format(i))
        results = train_model_vgg16(X_train_aug[i], X_val_aug[i], y_train_aug[i], y_val_aug[i], X_test, y_test,
                                       exp_name, results, hyperparams, i, model = None, pretrained= False)
        K.clear_session()
        tf.compat.v1.reset_default_graph()
        gc.collect()


else:
    for i in range(0, len(X_train_aug)):
        print("training_our_model_{}".format(i))
        results, hyperparams = train_model(X_train_aug[i], X_val_aug[i], y_train_aug[i], y_val_aug[i], X_test, y_test, exp_name,
                              results, hyperparams, i, lr=0.00001, lmbd=0.0001)
        K.clear_session()
        tf.compat.v1.reset_default_graph()
        gc.collect()
print('Results:', results, file=open('results/Results_{}_{}.txt'.format(exp_name, date), "w"))
print('Hyperparams:', hyperparams, file=open('results/Hyperparams{}_{}.txt'.format(exp_name, date), "w"))