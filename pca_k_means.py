
import seaborn
import numpy as np
import matplotlib.pyplot as plt
import os
import cv2
import time
import random
import tensorflow
import sklearn
import pandas as pd
from pathlib import Path
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

# sklearn
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical

from utils import create_training_data_k_means, reshape_and_normalize, scree_plot, fit_PCA, elbow_plot, fit_k_means, counter, plotter, plot_counts, plot_scatter


path = os.getcwd()
DATADIR = path+'/labeled_data'
DATADIR = Path(DATADIR)

#Sub directories for different categories
CATEGORIES = ["10_1","10_2","10_3"]

print('Path:', path)
print('Data directory:', DATADIR)

X, Y = create_training_data_k_means()

X_train, Y_train = reshape_and_normalize(X, Y, nb_classes=3)

pca, pca_fit, scores_pca = fit_PCA(X_train, n_components=10)
scree_plot(pca)

pca, pca_fit, scores_pca = fit_PCA(X_train, n_components=2)
PCA_components = pd.DataFrame(scores_pca)

elbow_plot(PCA_components)

kmeans, k_means_labels, Y_clust, unique_labels = fit_k_means(PCA_components, Y_train, number_of_clusters=3)

label_count= [[] for i in range(unique_labels)]
for n in range(unique_labels):

  label_count[n] = counter(Y_clust[n])
print(label_count) #Number of items of a certain category in cluster 1

plot_counts(label_count)

plot_scatter(k_means_labels, kmeans, PCA_components)

print('Reached end')