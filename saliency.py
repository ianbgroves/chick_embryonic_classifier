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

image_titles = [split_dict['y_test']] # 0 is 10(early) 1 is 10(late), or if wing, 0 is control, 1 is treated

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

