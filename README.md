# Code for the paper 'Bespoke data augmentation and network construction allow for image classification on small microscopy datasets'

# Installation:

1. Install [Anaconda](https://docs.anaconda.com/anaconda/install/index.html)
2. Clone the repository
3. From the command line, cd to the repository e.g. ```rb cd documents/github/chick_embryo_DCNN_classifier```
4. run ```rb conda create --name chick_dcnn_classifier python=3.6```
5. run ```rb conda activate chick_dcnn_classifier```
6. run ```rb pip install --no-deps -r requirements.txt``` to install all packages needed.

---
With the repository as the working directory:

# For clustering


```rb
python pca_k_means.py
```





# For neural network training

Use the following tags for the different augmentation regimes described in the paper.


*   ```rb --baseline``` for 'Baseline' 
*  ```rb --cutout``` for 'Cutout'
*   ```rb --shear``` for 'Shear
*   ```rb --gblur``` for 'Gaussian Blur'
*   ```rb --crop``` for 'Crop'
*   ```rb --randcomb``` for 'Random combination'
*   ```rb --mobius``` for 'Möbius transformations'


For InceptionV3, run ```rb python training_inceptionv3.py --baseline``` (e.g.)\
For ResNet50, run ```rb python training_resnet50.py --baseline``` (e.g.)\
For our model, run ```rb python training_our_model.py --baseline``` (e.g.)






