# Code for the paper 'Bespoke data augmentation and network construction enable developmental morphological classification on limited microscopy datasets'.

# Installation:

1. Install [Anaconda](https://docs.anaconda.com/anaconda/install/index.html)
2. Clone the repository
3. From the command line, cd to the repository e.g. ```cd documents/github/chick_embryo_DCNN_classifier```
4. run ```conda create --name chick_dcnn_classifier python=3.6```
5. run ```conda activate chick_dcnn_classifier```
6. run ```pip install --no-deps -r requirements.txt``` to install all packages needed.

---
With the repository as the working directory:

# Clustering (Figs 1-2)


```rb
python pca_k_means.py
```



# Neural network training (Tables 1-2, S2-5 tables)

Use the following tags for the different augmentation regimes described in the paper.


*   ```--baseline``` for 'Baseline' 
*  ```--cutout``` for 'Cutout'
*   ```--shear``` for 'Shear
*   ```--gblur``` for 'Gaussian Blur'
*   ```--crop``` for 'Crop'
*   ```--randcomb``` for 'Random combination'
*   ```--mobius``` for 'Möbius transformations'


For InceptionV3, run ``` python training_inceptionv3.py --baseline``` (e.g.)\
For ResNet50, run ``` python training_resnet50.py --baseline``` (e.g.)\
For our model, run ``` python training_our_model.py --baseline``` (e.g.)




# Saliency analysis (Figs 3-4, S4-5 Figs)

Figure 3 and S4 Fig

```rb
python saliency_maps_brain.py
```

Figure 4 and S5 Fig

```rb
python saliency_maps_limb.py
```


# Traditional classifiers (Support vector machine, K nearest neighbours, Random Forest classifier, S1 Table)

```rb
training_svm_rfc_knn.py
```
