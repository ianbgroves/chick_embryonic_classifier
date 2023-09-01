import numpy as np
import tensorflow as tf
import tensorflow.keras as keras
import os
import pickle as pkl  # module for serialization
import math

from matplotlib import pyplot as plt
from PIL import Image
from tensorflow.keras.applications.vgg16 import preprocess_input
from tf_keras_vis.utils.model_modifiers import ReplaceToLinear
from tf_keras_vis.utils.scores import CategoricalScore, BinaryScore
from tf_keras_vis.saliency import Saliency

from colab_utils import *

allcomb_path = "final_model/final_model"
allcomb_model = keras.models.load_model(allcomb_path)

split_dict = load_test_set("saved_test_sets/binary_baseline_3_Mar-14-2023/pkl_splits")

# plot the test set
images = np.array(split_dict["X_test"])

dims = math.ceil(np.sqrt(len(images)))

X = preprocess_input(images)
X = np.reshape(X, (-1, 200, 200, 1))
image_titles = []
for i in range(0, len(split_dict['y_test'])):
    image_titles.append(str(split_dict['y_test'][i]))

num_images = len(image_titles)
num_cols = min(num_images, 4)  # set the desired number of columns
num_rows = math.ceil(num_images / num_cols)

fig, ax = plt.subplots(nrows=num_rows, ncols=num_cols, figsize=(5, num_rows * 4))

for i, title in enumerate(image_titles):
    row = i // num_cols
    col = i % num_cols

    ax[row][col].set_title(title, fontsize=12)
    ax[row][col].imshow(X[i], cmap='gray')
plt.subplots_adjust(wspace=0.8, hspace=0.8)
plt.show()

## Saliency maps
images = np.array(split_dict["X_test"])

dims = math.ceil(np.sqrt(len(images)))

X = preprocess_input(images)
X = np.reshape(X, (-1, 200, 200, 1))

replace2linear = ReplaceToLinear()
print(split_dict["y_test"])
print(len(split_dict["y_test"]))

score = CategoricalScore(split_dict["y_test"])

print(tf.shape(X))  # Should be: <tf.Tensor: shape=(4,), dtype=int32, numpy=array([  3, 200, 200,   1], dtype=int32)>

model_list = [allcomb_model]

name_list = ['i']

image_titles = [split_dict['y_test']]  # 0 is 10(early) 1 is 10(late), or if wing, 0 is control, 1 is treated

image_titles = []

for i in range(0, len(split_dict['y_test'])):
    image_titles.append(str(split_dict['y_test'][i]))

for model in model_list:
    # Create Saliency object.
    saliency = Saliency(model,
                        model_modifier=replace2linear,
                        clone=True)

    # Create Saliency object.
    saliency_map = saliency(score,
                            X,
                            smooth_samples=20,  # The number of calculating gradients iterations.
                            smooth_noise=0.20)  # noise spread level.

    import math

    num_images = len(image_titles)
    num_cols = 4  # set the desired number of columns
    num_rows = math.ceil(num_images / num_cols)

    fig, ax = plt.subplots(nrows=num_rows, ncols=num_cols, figsize=(5, num_rows * 4))

    for i, title in enumerate(image_titles):
        row = i // num_cols
        col = i % num_cols

        ax[row][col].set_title(title, fontsize=12)
        ax[row][col].imshow(X[i])
        ax[row][col].imshow(saliency_map[i], cmap='jet', alpha=0.4)
        ax[row][col].axis('off')

    # remove empty subplots
    for i in range(num_rows):
        for j in range(num_cols):
            if i * num_cols + j >= num_images:
                fig.delaxes(ax[i][j])

    plt.tight_layout()
    plt.subplots_adjust(wspace=0.8, hspace=0.8)
    plt.show()

# below code is used to
# calculate the average saliency maps

os.chdir('data_10_early_late_aligned')
data = create_data(os.path.join(os.getcwd(), ''), duplicate_channels=False)

data_list = []
data_list.append(data[0:len(data)])

X = []
Y = []

for i in data_list:
    for feature, label in i:
        X.append(feature)
        Y.append(label)

# plot the aligned images
print('plotting aligned images for average saliency')

images = np.array(X)

dims = math.ceil(np.sqrt(len(images)))

X = np.reshape(X, (-1, 200, 200, 1))
image_titles = []
for i in range(0, len(Y)):
    image_titles.append(str(Y[i]))

num_images = len(image_titles)
num_cols = 3  # set the desired number of columns
num_rows = math.ceil(num_images / num_cols)

fig, ax = plt.subplots(nrows=num_rows, ncols=num_cols, figsize=(5, num_rows * 4))

for i, title in enumerate(image_titles):
    row = i // num_cols
    col = i % num_cols

    ax[row][col].set_title(title, fontsize=12)
    ax[row][col].imshow(X[i], cmap='gray')
plt.subplots_adjust(wspace=0.8, hspace=0.8)
plt.show()

# reshaping the images for tensorflow/tf-keras-vis
images_registered = np.array(X)
images_registered_reshaped = np.reshape(images_registered, (-1, 200, 200))

images = np.array(X)

X = preprocess_input(images_registered_reshaped)
X = np.reshape(X, (-1, 200, 200, 1))

replace2linear = ReplaceToLinear()

score = CategoricalScore(Y)

model_list = [allcomb_model]
name_list = ['allcomb']

# separating out the images (X) and the labels (Y) into early and late
X_0 = X[0:6]
X_1 = X[7:]
Y_0 = Y[0:6]
Y_1 = Y[7:]


def plot_average_saliency_map(X, Y, model_list, name_list, early, late):
    all_saliency_maps = []
    image_titles = []
    # Iterate through the different models, generating a saliency map for each

    score_0 = CategoricalScore(Y)

    print(
        tf.shape(X))  # Should be: <tf.Tensor: shape=(4,), dtype=int32, numpy=array([  3, 200, 200,   1], dtype=int32)>

    for i in range(0, len(Y)):
        image_titles.append(str(Y[i]))

    for model, name in zip(model_list, name_list):
        # Create Saliency object.
        saliency = Saliency(model,
                            model_modifier=replace2linear,
                            clone=True)

        # Create Saliency object.
        saliency_map = saliency(score_0,
                                X,
                                smooth_samples=20,
                                smooth_noise=0.20)

        all_saliency_maps.append(saliency_map)

    # Get the saliency maps for the selected model
    selected_model_saliency_maps = all_saliency_maps[0]

    # Calculate the average saliency map across all images for the selected model
    average_saliency_map = np.mean(selected_model_saliency_maps, axis=0)

    # Visualize the average saliency map for the selected model
    if early:
        print('displaying average saliency map for the 10 (early) class')
        plt.imshow(X[1])  # this is the image that all the 10 (early) test images were aligned to
    elif late:
        print('displaying average saliency map for the 10 (late) class')
        plt.imshow(X[0])  # this is the image that all the 10 (late) test images were aligned to
    plt.imshow(average_saliency_map, cmap='jet', alpha=0.4)
    plt.show()


plot_average_saliency_map(X_0, Y_0, model_list, name_list, early=True, late=False)
plot_average_saliency_map(X_1, Y_1, model_list, name_list, early=False, late=True)
