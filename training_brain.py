from datetime import date as dt
import os
import gc
import pickle as pkl  # module for serialization
from tensorflow.keras import backend as k
from colab_utils import *

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
X_train_aug, y_train_aug, X_val_aug, y_val_aug = augment_data(X_train, y_train, X_val, y_val, baseline, cutout, shear,
                                                              gblur, crop, randcomb, mobius, allcomb_sparse,
                                                              allcomb_full, resnet, inception, limb=False)

# Set up dictionaries for logging.
results = {"accuracies": [], "losses": [], "val_accuracies": [],
           "val_losses": [], "test_performance": [], "test_accuracies": [], "test_losses": []}
hyperparams = {"configuration": [], "loss_func": [], "optimizer": [], "learning_rate": [], "lambda": []}

# In the paper- we finetuned ResNet50/InceptionV3
# From scratch-training takes a long time and careful management of the learning rate,
# pass either your already trained model or set pretrained=True

if resnet or inception:
    results, hyperparams = finetune_resnet_inception(X_train_aug, X_val_aug, y_train_aug, y_val_aug, X_test, y_test,
                                  exp_name, results, hyperparams, model=None, pretrained=True, freeze=True, resnet=True, inception=False)

# If not using Resnet/Inception, 'our model' in the paper.
else:

    for i in range(0, len(X_train_aug)):
        print("training_our_model_{}".format(i))
        results, hyperparams = train_model(X_train_aug[i], X_val_aug[i], y_train_aug[i], y_val_aug[i], X_test, y_test,
                                           exp_name,
                                           results, hyperparams, i, lr=0.00001, lmbd=0.0001)
        k.clear_session()
        tf.compat.v1.reset_default_graph()
        gc.collect()

print('Results:', results, file=open('results/Results_{}_{}.txt'.format(exp_name, date), "w"))
print('Hyperparams:', hyperparams, file=open('results/Hyperparams{}_{}.txt'.format(exp_name, date), "w"))
