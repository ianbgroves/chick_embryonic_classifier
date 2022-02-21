
import numpy as np
from matplotlib import pyplot as plt
import tensorflow as tf
import tensorflow.keras as keras
from PIL import Image
from tensorflow.keras.preprocessing.image import load_img
from tensorflow.keras.applications.vgg16 import preprocess_input
from tf_keras_vis.utils.model_modifiers import ReplaceToLinear
from tf_keras_vis.utils.scores import CategoricalScore
from tensorflow.keras import backend as K
from tf_keras_vis.saliency import Saliency

path = os.getcwd()

baseline_path = 'baseline3_8Nov-17-2021'
crop_path = 'crop3_8Nov-17-2021'
cutout_path = 'cutout3_8Nov-17-2021'
shear_path = 'shear4_8Nov-17-2021'
blur_path = 'gblur2_8Nov-17-2021'
rand_comb_path = 'rand_cutout_gblur_shear2_8Nov-18-2021'

baseline_model = keras.models.load_model(baseline_path)
crop_model = keras.models.load_model(crop_path)
cutout_model = keras.models.load_model(cutout_path)
shear_model = keras.models.load_model(shear_path)
blur_model = keras.models.load_model(blur_path)
rand_comb_model = keras.models.load_model(rand_comb_path)

# Image titles
image_titles = ['10.1', '10.2', '10.3']

# Load images and Convert them to a Numpy array

img1 = Image.open('10_1_b.png').convert('L')
img1 = img1.resize((200,200), Image.ANTIALIAS)
img1 = np.array(img1)

img2 = Image.open('10_2_a.png').convert('L')
img2 = img2.resize((200,200), Image.ANTIALIAS)
img2 = np.array(img2)

img3 = Image.open('10_3_b.png').convert('L')
img3 = img3.resize((200,200), Image.ANTIALIAS)
img3 = np.array(img3)

images = np.asarray([np.array(img1), np.array(img2), np.array(img3)])

# Preparing input data for VGG16 style network
X = preprocess_input(images)
X = np.reshape(X,(-1, 200,200, 1))

# Rendering

for i, title in enumerate(image_titles):
    plt.figure()
    plt.imshow(images[i])
    plt.savefig('{}.svg'.format(image_titles[i]))

replace2linear = ReplaceToLinear()


# 1 is the imagenet index corresponding to Goldfish, 294 to Bear and 413 to Assault Rifle.

score = CategoricalScore([0, 1, 2])
def score_function(output):
    # The `output` variable refers to the output of the model,
    # so, return the three values for the 1st, the 2nd and the 3rd of sub-stages respectively.
   return (output[None, 0], output[None, 1], output[None, 2])

   
tf.shape(X)  # Should be: <tf.Tensor: shape=(4,), dtype=int32, numpy=array([  3, 200, 200,   1], dtype=int32)>

model_list = [baseline_model, crop_model, cutout_model, shear_model, blur_model, rand_comb_model]
name_list = ['baseline', 'crop', 'cutout', 'shear', 'blur', 'randcomb']

j = 0
for model in model_list:
  
  # Create Saliency object.
  saliency = Saliency(model,
                      model_modifier=replace2linear,
                      clone=True)
  # Generate saliency map with smoothing that reduce noise by adding noise
  saliency_map = saliency(score,
                          X,
                          smooth_samples=20, # The number of calculating gradients iterations.
                          smooth_noise=0.05) # noise spread level.


  # Render
  for i, title in enumerate(image_titles):
      plt.figure()
      plt.imshow(saliency_map[i], cmap='jet')
      plt.savefig('{}_saliency.svg'.format(image_titles[i]))     
 
j = j+1