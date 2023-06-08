# Code for the paper 'Accurate staging of chick embryonic tissues via deep learning'.

[Here](https://colab.research.google.com/drive/1wH53iao1chYqNcCUbx7cXNvgCMm0gRFT?usp=sharing) is the Colab notebook where you can train DCNNs on your own data 


# Installation:

1. Install [Anaconda](https://docs.anaconda.com/anaconda/install/index.html)
2. Clone the repository
3. From the command line, cd to the repository e.g. ```cd documents/github/chick_embryo_DCNN_classifier```
4. run ```conda create --name chick_dcnn_classifier python=3.6```
5. run ```conda activate chick_dcnn_classifier```
6. run ```pip install --no-deps -r requirements.txt``` to install all packages needed.

---
With the repository as the working directory:


# Neural network training (Tables 1, S2-3 tables)

Use the following tags for the different augmentation regimes described in the paper.


*   ```--baseline``` for 'Baseline' 
*  ```--cutout``` for 'Cutout'
*   ```--shear``` for 'Shear
*   ```--gblur``` for 'Gaussian Blur'
*   ```--crop``` for 'Crop'
*   ```--randcomb``` for '1+2/4/5 RC in Table 1'
*   ```--allcomb``` for '1+2+4+5 RC in Table 1'
*   ```--mobius``` for 'Möbius transformations'


For our model, run ``` python training_our_model.py --baseline``` (e.g.)

Or, for InceptionV3/ResNet50 re-training, set either the resnet/inception booleans to true in ``` training_brain.py ```\
and then run ``` python training_brain.py --baseline``` (e.g.)


For the unsupervised clustering in 
Fig S1
run
```rb
python pca_k_means.py
```
For the unsupervised clustering on Haralick features run
Fig S2
```rb
python feature_extraction.py
```



# Traditional classifiers (Support vector machine, K nearest neighbours, Random Forest classifier, S1 Table)
run 
```rb
training_svm_rfc_knn.py
```
