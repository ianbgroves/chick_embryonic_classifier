# Code for the paper 'Bespoke data augmentation and network construction allow for image classification on small microscopy datasets'

# Installation:

1. Install [Anaconda](https://docs.anaconda.com/anaconda/install/index.html)
2. Clone the repository
3. From the command line, cd to the repository e.g. cd documents/github/chick_embryo_DCNN_classifier
4. run conda create --name chick_dcnn_classifier python=3.6
5. run pip install -r requirements.txt to install all dependencies

---
With the repository as the working directory:

# For clustering


*  run '''rb python pca_k_means.py'''





# For neural network training

Use the following tags for the different augmentation regimes described in the paper.


*   --baseline for 'Baseline' 
*   --cutout for 'Cutout'
*   --shear for 'Shear
*   --gblur for 'Gaussian Blur'
*   --crop for 'Crop'
*   --randcomb for 'Random combination'
*   --mobius for 'Möbius transformations'


For InceptionV3, run python training_inceptionv3.py --baseline (e.g.)\
For ResNet50, run python training_resnet50.py --baseline (e.g.)\
For our model, run python training_our_model.py --baseline (e.g.)






