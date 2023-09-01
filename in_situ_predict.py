import numpy as np
from matplotlib import pyplot as plt
import tensorflow as tf
import tensorflow.keras as keras
import pickle as pkl # module for serialization
import math
from colab_utils import *


allcomb_path = "final_model/final_model"
allcomb_model = keras.models.load_model(allcomb_path)

split_dict = load_test_set("saved_test_sets/binary_baseline_3_Mar-14-2023/pkl_splits")


path = 'data_in_situ'
data = create_data(path, duplicate_channels=False)

print(len(data))
# should be 11
data_list = []
data_list.append(data[0:len(data)])

X = []
Y = []
for i in data_list:
    for feature, label in i:
        X.append(feature)
        Y.append(label)

X_np = np.array(X)
Y_np = np.array(Y)

X_np = X_np / 255
X_np = X_np.reshape(X_np.shape[0], 200, 200, 1)
Y_np = to_categorical(Y_np)

# print('X shape:', np.shape(X_np))  # should be (number of test images, 200, 200, 1)
# print('Y shape', np.shape(Y_np))  # should be (number of test labels, 2)


scores = allcomb_model.evaluate(X_np, Y_np)

predicted_labels = allcomb_model.predict(X_np)

predicted_label_class = np.argmax(predicted_labels, axis=1)
print("predicted_classes, 0: 10(early), 1: 10(late")
print(predicted_label_class)