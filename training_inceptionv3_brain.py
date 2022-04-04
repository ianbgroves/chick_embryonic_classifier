import os
import cv2
from pathlib import Path
from random import seed, shuffle
import numpy as np
from imgaug import augmenters as iaa

from utils import k_fold_splitter, create_training_data, aug_rot, aug_crop, aug_cutout, aug_shear, aug_g_blur, \
    aug_rand_comb, train_model_inceptionv3, read_args, aug_mobius, shift_func, mobius_fast_interpolation, \
    getabcd_1fix, M_admissable, madmissable_abcd

# Initialise booleans parsed from the command line

baseline = False
cutout = False
shear = False
g_blur = False
crop = False
rand_comb = False
mobius = False

# Read from the command line which augmentation to apply
baseline, cutout, shear, g_blur, crop, rand_comb, mobius = read_args(baseline=False, cutout=False,
                                                                     shear=False, g_blur=False, crop=False,
                                                                     rand_comb=False, mobius=False)

# Define paths for the dataset and the classes

path = os.getcwd()
DATADIR = path + '/labeled_data'
DATADIR = Path(DATADIR)
SUBSTAGES = ["10_1", "10_2", "10_3"]
print('Path:', path)
print('Data directory:', DATADIR)

# Define paths to save the model and/or training plots
parent = os.path.join(path, os.pardir)
model_dir = os.path.abspath(parent) + '/Models'
plot_DIR = os.path.abspath(parent) + '/Plots/augmentation_acc_loss_plots'

print("\nParent Directory:", os.path.abspath(parent))
print("Model directory:", model_dir)
print("Plots directory:", plot_DIR)

# Displays number of images - should be 151
image_count = len(list(DATADIR.glob('*/*.jpg')))
print(image_count)

# Create and shuffle the dataset
# ResNet50 requires RGB images
# Read data in as grayscale ("L"), and duplicate the channels to emulate RGB
t_data = create_training_data(imformat="L", duplicate_channels=True)
seed(123)
shuffle(t_data)

# k fold cross-validation
# loop through the dataset, splitting it into equal partitions
k = 10
split_data = []
for i in range(0, k):
    split_data.append(t_data[int(round(len(t_data) * (i / k), 0)):int(round(len(t_data) * ((i + 1) / k), 0))])

# augment the dataset with the chosen augmentation
# split the augmented data into k folds
split_aug_data = []  # initialise array to contain augmented data
if baseline:
    for i in range(0, k):
        split_aug_data.append(aug_rot(split_data[i]))
    val_list, training_list = k_fold_splitter(split_aug_data, k)

if cutout:
    for i in range(0, k):
        split_aug_data.append(aug_cutout(split_data[i]))
    val_list, training_list = k_fold_splitter(split_aug_data, k)

if shear:
    for i in range(0, k):
        split_aug_data.append(aug_shear(split_data[i]))
    val_list, training_list = k_fold_splitter(split_aug_data, k)

if g_blur:
    for i in range(0, k):
        split_aug_data.append(aug_g_blur(split_data[i]))
    val_list, training_list = k_fold_splitter(split_aug_data, k)

if crop:
    for i in range(0, k):
        split_aug_data.append(aug_crop(split_data[i]))
    val_list, training_list = k_fold_splitter(split_aug_data, k)

if rand_comb:
    for i in range(0, k):
        split_aug_data.append(aug_rand_comb(split_data[i]))
    val_list, training_list = k_fold_splitter(split_aug_data, k)

if mobius:
    split_mobius_data = []

    # mobius method uses RGB images, and then converts to grayscale as the final step

    t_data = create_training_data(imformat="RGB", duplicate_channels=False)
    seed(123)
    shuffle(t_data)
    k = 10
    split_data = []

    # split the image data, augment it with mobius transformations, then rotations
    for i in range(0, k):
        split_data.append(t_data[int(round(len(t_data) * (i / k), 0)):int(round(len(t_data) * ((i + 1) / k), 0))])
        split_aug_data.append(aug_mobius(split_data[i], M=2, mode='wrap', user_defined=False,
                                         rgb=True))  # M must be > 1, and this is slower the closer to that
        split_mobius_data.append(aug_rot(split_aug_data[i]))

    val_list, training_list = k_fold_splitter(split_mobius_data, k)


val_accs = []  # Initialise variable where validation accuracies will be saved
save_name = 'test'

# Iterate training, using a different training and validation set for each model
for i in range(0, len(training_list)):
    print('Training model {}'.format(i))
    val_accs.append(train_model_inceptionv3(training_list[i], val_list[i], save_name + '{}'.format(i)))
    print('Model {} validation accuracy {}'.format(i, val_accs[i]))
print('Validation accuracies:', val_accs, file=open('val_acc_output_{}.txt'.format(save_name), "w"))
print('Average validation accuracy:', round(np.mean(val_accs), 1),
      file=open('val_acc_avg_output_{}.txt'.format(save_name), "w"))
print('Validation standard deviation:', round(np.std(val_accs), 1),
      file=open('val_acc_std_output_{}.txt'.format(save_name), "w"))
print('=========================================')

print("Reached end")
