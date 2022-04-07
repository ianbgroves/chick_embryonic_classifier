import numpy as np
from matplotlib import pyplot as plt
import tensorflow as tf
import tensorflow.keras as keras
from PIL import Image
import os
from tensorflow.keras.preprocessing.image import load_img
from tensorflow.keras.applications.vgg16 import preprocess_input
from tf_keras_vis.utils.model_modifiers import ReplaceToLinear
from tf_keras_vis.utils.scores import CategoricalScore
from tf_keras_vis.saliency import Saliency
from tensorflow.keras import backend as K

path = os.getcwd()

baseline_path = 'Models/limb/limb_test_baseline_flip_w_test_heldback8Mar-17-2022'
crop_path = 'Models/limb/limb_test_crop_w_test_heldback8Mar-17-2022'
cutout_path = 'Models/limb/limb_test_cutout_w_test_heldback8Mar-17-2022'
shear_path = 'Models/limb/limb_test_gblur_w_test_heldback8Mar-17-2022'
blur_path = 'Models/limb/limb_test_randcomb_w_test_heldback8Mar-17-2022'
rand_comb_path = 'Models/limb/limb_test_shear_w_test_heldback8Mar-17-2022'

baseline_model = keras.models.load_model(baseline_path)
crop_model = keras.models.load_model(crop_path)
cutout_model = keras.models.load_model(cutout_path)
shear_model = keras.models.load_model(shear_path)
blur_model = keras.models.load_model(blur_path)
rand_comb_model = keras.models.load_model(rand_comb_path)

# Image titles
fig4_image_titles = ['Fig4_A', 'Fig4_B']  # control and treated respectively
s5_fig_image_titles = ['S5_Fig_A_left', 'S5_Fig_A_right']  # control and treated respectively

# Load images and Convert them to a Numpy array

img1 = Image.open('saliency_tests/limb/c_ctrl.jpg').convert('L')
img1 = img1.resize((200,200), Image.ANTIALIAS)
img1 = np.array(img1)

img2 = Image.open('saliency_tests/limb/c_treated.jpg').convert('L')
img2 = img2.resize((200,200), Image.ANTIALIAS)
img2 = np.array(img2)

img3 = Image.open('saliency_tests/limb/d_ctrl.jpg').convert('L')
img3 = img3.resize((200,200), Image.ANTIALIAS)
img3 = np.array(img3)

img4 = Image.open('saliency_tests/limb/d_treated.jpg').convert('L')
img4 = img4.resize((200,200), Image.ANTIALIAS)
img4 = np.array(img4)


fig4_images = np.asarray([np.array(img1), np.array(img2)])
s5_fig_images = np.asarray([np.array(img3), np.array(img4)])


# Preparing input data for VGG16 style network
fig4_processed = preprocess_input(fig4_images)
fig4_processed = np.reshape(fig4_processed, (-1, 200,200, 1))

s5_fig_processed = preprocess_input(s5_fig_images)
s5_fig_processed = np.reshape(s5_fig_processed, (-1, 200,200, 1))


# Rendering

for i, title in enumerate(fig4_image_titles):
    plt.figure()
    plt.imshow(fig4_images[i], cmap='gray')
    plt.savefig('{}.svg'.format(fig4_image_titles[i]))


for i, title in enumerate(s5_fig_image_titles):
    plt.figure()
    plt.imshow(s5_fig_images[i], cmap='gray')
    plt.savefig('{}.svg'.format(s5_fig_image_titles[i]))


replace2linear = ReplaceToLinear()


score = CategoricalScore([0, 1])
def score_function(output):
    # The `output` variable refers to the output of the model,
    # so, return the two values for the control and treated respectively.
   return (output[None, 0], output[None, 1])

   
tf.shape(fig4_processed)  # Should be: <tf.Tensor: shape=(4,), dtype=int32, numpy=array([  3, 200, 200,   1], dtype=int32)>

model_list = [baseline_model,  blur_model, cutout_model, shear_model, rand_comb_model, crop_model]
name_list = ['i', 'ii', 'iii','iv', 'v', 'vi']
j = 0

for j, model in zip(name_list, model_list):

    # Create Saliency object.
    saliency = Saliency(model,
                        model_modifier=replace2linear,
                        clone=True)
    # Generate saliency map with smoothing that reduce noise by adding noise
    saliency_map = saliency(score,
                            fig4_processed,
                            smooth_samples=20,  # The number of calculating gradients iterations.
                            smooth_noise=0.05)  # noise spread level.


    # Render
    f, ax = plt.subplots(nrows=1, ncols=2, figsize=(12, 4), dpi=300)

    for i, title in enumerate(fig4_image_titles):
        ax[i].set_title(title, fontsize=14)
        ax[i].imshow(saliency_map[i], cmap='jet')
        ax[i].axis('off')

    plt.tight_layout()

    f.savefig(
        'Saliency_outputs/limb/Fig4_panel_{}.svg'.format(
            j), bbox_inches='tight')

    




# Create Saliency object.
s5_saliency = Saliency(rand_comb_model,
                    model_modifier=replace2linear,
                    clone=True)
# Generate saliency map with smoothing that reduce noise by adding noise
s5_saliency_map = s5_saliency(score,
                        s5_fig_processed,
                        smooth_samples=20,  # The number of calculating gradients iterations.
                        smooth_noise=0.05)  # noise spread level.


# Render
f, ax = plt.subplots(nrows=1, ncols=2, figsize=(12, 4), dpi=300)

for i, title in enumerate(s5_fig_image_titles):
    ax[i].set_title(title, fontsize=14)
    ax[i].imshow(s5_saliency_map[i], cmap='jet')
    ax[i].axis('off')

plt.tight_layout()

f.savefig(
    'Saliency_outputs/limb/S5_Fig_panel_{}.svg'.format(
        j), bbox_inches='tight')

    
# plt.show()