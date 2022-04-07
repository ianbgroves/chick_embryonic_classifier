

import os
import cv2
from pathlib import Path
from random import seed, shuffle
import numpy as np
from imgaug import augmenters as iaa

from utils import k_fold_splitter, create_training_data, aug_rot, aug_crop, aug_cutout, aug_shear, aug_gblur, aug_randcomb, train_model, read_args, aug_mobius, shift_func, mobius_fast_interpolation, getabcd_1fix, M_admissable, madmissable_abcd

baseline = False
cutout = False
shear = False
gblur = False
crop = False
randcomb = False
mobius = False

path = os.getcwd()
DATADIR = path+'/labeled_data'
DATADIR = Path(DATADIR)

#Sub directories for different categories
CATEGORIES = ["10_1","10_2","10_3"]

print('Path:', path)
print('Data directory:', DATADIR)


parent = os.path.join(path, os.pardir)
modelDIR = os.path.abspath(parent) + '/Models'
plotDIR = os.path.abspath(parent) + '/Plots/augmentation_acc_loss_plots'
imagetestdir = os.path.abspath(parent) + '/Image_test_dir'
# prints parent directory
print("\nParent Directory:", os.path.abspath(parent))
print("Model directory:", modelDIR)
print("Plots directory:", plotDIR)
print("image test directory:", imagetestdir)

image = cv2.imread('labeled_data/10_1/10.1_035.jpg', cv2.IMREAD_GRAYSCALE)
# print(image)
image_count = len(list(DATADIR.glob('*/*.jpg')))
print(image_count)


baseline, cutout, shear, gblur, crop, randcomb, mobius = read_args(baseline=False, cutout=False, shear=False, gblur=False, crop=False, randcomb=False, mobius=False)

tdata = create_training_data(imformat="L", duplicate_channels=False)
seed(123)
shuffle(tdata)
k=10
split_data = []
for i in range(0, k):
  split_data.append(tdata[int(round(len(tdata)*((i)/k), 0)):int(round(len(tdata)*((i+1)/k), 0))])


split_aug_data = []
split_aug_cutout = []
split_mobius_data = []
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

if gblur:

  for i in range(0, k):
    split_aug_data.append(aug_gblur(split_data[i]))   
  val_list, training_list = k_fold_splitter(split_aug_data, k)

if crop:

  for i in range(0, k):
    split_aug_data.append(aug_crop(split_data[i]))   
  val_list, training_list = k_fold_splitter(split_aug_data, k)

if randcomb:

  for i in range(0, k):
    split_aug_data.append(aug_randcomb(split_data[i]))   
  val_list, training_list = k_fold_splitter(split_aug_data, k)

if mobius:

  training_list = []
  val_list = []

  # mobius method uses RGB images, and then converts to grayscale as the final step

  tdata = create_training_data(imformat="L", duplicate_channels=True)
  seed(123)
  shuffle(tdata)
  k=10
  split_data = []

  for i in range(0, k):
    split_data.append(tdata[int(round(len(tdata)*((i)/k), 0)):int(round(len(tdata)*((i+1)/k), 0))])
    split_aug_data.append(aug_mobius(split_data[i], M=2, mode='wrap', user_defined=False, rgb=False))   # M must be > 1, and this is slower the closer to that
    split_mobius_data.append(aug_rot(split_aug_data[i]))
  
  val_list, training_list = k_fold_splitter(split_mobius_data, k)
  # for i in range(0, k):    
  #   val_list.append(np.array(split_mobius_data[i]))
  #   training_list.append(np.delete(split_mobius_data, i))


valaccs = []
savename = 'test'

for i in range(0, len(training_list)):
    print('Training model {}'.format(i))
    valaccs.append(train_model(training_list[i], val_list[i], savename + '{}'.format(i)))
    print('Model {} validation accuracy {}'.format(i, valaccs[i]))
print('Validation accuracies:', valaccs,  file=open('valacc_output_{}.txt'.format(savename), "w"))
print('Average validation accuracy:', round(np.mean(valaccs), 1), file=open('valaccavg_output_{}.txt'.format(savename), "w"))
print('Validation standard deviation:', round(np.std(valaccs), 1), file=open('valaccstd_output_{}.txt'.format(savename), "w"))
print('=========================================')

print("Reached end")

