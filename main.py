!pip install -U folium==0.2.1
!pip install -U packaging==20.9
!pip install -U opencv-contrib-python
!pip install git+https://github.com/rdk2132/scikeras
!pip install hypopt
!pip install tensorflowjs
!pip install patool

import os
import random
import struct
import time
import io
import zipfile
import requests

import cv2
from google.colab.output import eval_js
from google.colab.patches import cv2_imshow
from google.colab import files

import keras.api.keras as keras
from keras.preprocessing.image import ImageDataGenerator
from keras import backend as K
from keras.models import Sequential, Model
from keras.layers import (Dense, Conv2D, Activation, MaxPooling2D, Dropout,
                          Flatten, Reshape, Input, BatchNormalization,
                          LeakyReLU, ZeroPadding2D, UpSampling2D,
                          Conv2DTranspose, Lambda, concatenate)
from keras.applications.mobilenet import MobileNet

import matplotlib.pyplot as plt
import matplotlib
import seaborn as sns
from PIL import Image

import gdown
import argparse
import numpy as np
import pandas as pd

from tqdm.notebook import tqdm

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn import tree
from sklearn.metrics import (confusion_matrix, classification_report,
                             precision_recall_curve, roc_auc_score,
                             make_scorer, accuracy_score)
from sklearn.base import BaseEstimator

from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis

from hypopt import GridSearch

from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

from sklearn.cluster import KMeans, AgglomerativeClustering

import tensorflowjs as tfjs
import tensorflow as tf

import patoolib
import os.path
from os import path
from distutils.dir_util import copy_tree

from ipywidgets import interact
import ipywidgets as widgets

start_time = time.time()

DATA_ROOT = '/content/data'
os.makedirs(DATA_ROOT, exist_ok=True)

metadata_url = 'https://drive.google.com/uc?id=1kmpa-Lnra-8KhEjj8r3nj9y4e53qcPGX'
metadata_path = os.path.join(DATA_ROOT, 'metadata.csv')
gdown.download(metadata_url, metadata_path, True)

images_1_url = 'https://drive.google.com/uc?id=1HW5HbQ_OR7xUPWfw4yg1r_EbhDaCuOrj'
image_path_1 = os.path.join(DATA_ROOT, 'images_1.zip')
gdown.download(images_1_url, image_path_1, True)

images_2_url = 'https://drive.google.com/uc?id=1VAbEVMKZmKyh9tpe9iVZ_XpTPLpxitwt'
image_path_2 = os.path.join(DATA_ROOT, 'images_2.zip')
gdown.download(images_2_url, image_path_2, True)

images_rgb_path = os.path.join(DATA_ROOT, 'hmnist_8_8_RGB.csv')

if not path.exists(os.path.join(DATA_ROOT, 'images_1')):
    patoolib.extract_archive(os.path.join(DATA_ROOT, 'images_1.zip'),
                             outdir=os.path.join(DATA_ROOT, 'images_1'))

if not path.exists(os.path.join(DATA_ROOT, 'images_2')):
    patoolib.extract_archive(os.path.join(DATA_ROOT, 'images_2.zip'),
                             outdir=os.path.join(DATA_ROOT, 'images_2'))

fromDirectory = os.path.join(DATA_ROOT, 'images_1')
toDirectory = os.path.join(DATA_ROOT, 'images_all')
copy_tree(fromDirectory, toDirectory)

fromDirectory = os.path.join(DATA_ROOT, 'images_2')
toDirectory = os.path.join(DATA_ROOT, 'images_all')
copy_tree(fromDirectory, toDirectory)

IMG_WIDTH = 100
IMG_HEIGHT = 75

metadata = pd.read_csv(metadata_path)
metadata['category'] = metadata['dx'].replace(
    {'akiec': 0, 'bcc': 1, 'bkl': 2, 'df': 3, 'mel': 4, 'nv': 5, 'vasc': 6}
)

X = []
X_g = []
y = []

for i in tqdm(range(len(metadata))):
    image_meta = metadata.iloc[i]
    img_path = os.path.join(toDirectory, image_meta['image_id'] + '.jpg')
    img = cv2.imread(img_path, cv2.IMREAD_COLOR)
    img = cv2.resize(img, (IMG_WIDTH, IMG_HEIGHT))
    img_g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    X_g.append(img_g)
    X.append(img)
    y.append(image_meta['category'])

X_g = np.array(X_g)
X = np.array(X)
y = np.array(y)

sample_cap = 142
option = 1

if option == 1:
    class_totals = [0] * 7
    iter_samples = [0] * 7
    indices = []

    for i in range(len(X)):
        class_totals[y[i]] += 1

    for i in range(len(X)):
        if iter_samples[y[i]] < sample_cap:
            indices.append(i)
            iter_samples[y[i]] += 1

    X = X[indices]
    X_g = X_g[indices]
    y = y[indices]

elif option == 2:
    class_totals = [0] * 7
    for i in range(len(X)):
        class_totals[y[i]] += 1

    largest_index = class_totals.index(max(class_totals))
    class_totals[largest_index] = 0
    second_largest_val = max(class_totals)

    indices = []
    count = 0
    for i in range(len(X)):
        if y[i] == largest_index:
            if count < second_largest_val:
                indices.append(i)
                count += 1
        else:
            indices.append(i)

    X = X[indices]
    X_g = X_g[indices]
    y = y[indices]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.4, random_state=101
)
X_g_train, X_g_test, y_train, y_test = train_test_split(
    X_g, y, test_size=0.4, random_state=101
)

X_augmented = []
X_g_augmented = []
y_augmented = []

for i in tqdm(range(len(X_train))):
    transform = random.randint(0, 1)
    if transform == 0:
        X_augmented.append(cv2.flip(X_train[i], 1))
        X_g_augmented.append(cv2.flip(X_g_train[i], 1))
        y_augmented.append(y_train[i])
    else:
        zoom = 0.33
        centerX, centerY = int(IMG_HEIGHT / 2), int(IMG_WIDTH / 2)
        radiusX, radiusY = int((1 - zoom) * IMG_HEIGHT * 2), int((1 - zoom) * IMG_WIDTH * 2)
        minX, maxX = centerX - radiusX, centerX + radiusX
        minY, maxY = centerY - radiusY, centerY + radiusY

        cropped = X_train[i][minX:maxX, minY:maxY]
        new_img = cv2.resize(cropped, (IMG_WIDTH, IMG_HEIGHT))
        X_augmented.append(new_img)

        cropped_g = X_g_train[i][minX:maxX, minY:maxY]
        new_img_g = cv2.resize(cropped_g, (IMG_WIDTH, IMG_HEIGHT))
        X_g_augmented.append(new_img_g)

        y_augmented.append(y_train[i])

X_augmented = np.array(X_augmented)
X_g_augmented = np.array(X_g_augmented)
y_augmented = np.array(y_augmented)

X_train = np.vstack((X_train, X_augmented))
X_g_train = np.vstack((X_g_train, X_g_augmented))
y_train = np.append(y_train, y_augmented)

classes = ['akiec', 'bcc', 'bkl', 'df', 'mel', 'nv', 'vasc']
colors = ['red', 'green', 'blue', 'purple', 'black', 'brown', 'orange']

X_flat = X.reshape(X.shape[0], X.shape[1] * X.shape[2] * X.shape[3])
X_g_flat = X_g.reshape(X_g.shape[0], X_g.shape[1] * X_g.shape[2])

pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_flat)

plt.scatter(X_pca[:, 0], X_pca[:, 1], c=y, cmap=matplotlib.colors.ListedColormap(colors))
cb = plt.colorbar()
loc = np.arange(0, max(y), max(y) / float(len(colors)))
cb.set_ticks(loc)
cb.set_ticklabels(classes)
plt.title("PCA Representation")
plt.show()

def plot_visualization(X_vis, y_vis, colors_vis, classes_vis, name):
    plt.scatter(X_vis[:, 0], X_vis[:, 1], c=y_vis,
                cmap=matplotlib.colors.ListedColormap(colors_vis))
    cb = plt.colorbar()
    loc = np.arange(0, max(y_vis), max(y_vis) / float(len(colors_vis)))
    cb.set_ticks(loc)
    cb.set_ticklabels(classes_vis)
    plt.title(name)
    plt.show()

tsne = TSNE(n_components=2, init="random", learning_rate=200.0)
X_tsne = tsne.fit_transform(X_flat)

plt.scatter(X_tsne[:, 0], X_tsne[:, 1], c=y,
            cmap=matplotlib.colors.ListedColormap(colors))
cb = plt.colorbar()
loc = np.arange(0, max(y), max(y) / float(len(colors)))
cb.set_ticks(loc)
cb.set_ticklabels(classes)
plt.title("T-SNE Representation")
plt.show()

lesion_img = X_g[0]

sift = cv2.xfeatures2d.SIFT_create()
keypoints, descriptor = sift.detectAndCompute(lesion_img, None)
sift_img = cv2.drawKeypoints(X[0], keypoints, lesion_img)
cv2_imshow(sift_img)

all_descriptors = []
for i in tqdm(range(X_train.shape[0])):
    kp, des = sift.detectAndCompute(X_train[i], None)
    if des is not None:
        for d in des:
            all_descriptors.append(d)

k = len(classes) * 10
sift_kmeans = KMeans(n_clusters=k)
sift_kmeans.fit(all_descriptors)

X_sift_train = []
y_sift_train = []

for i in tqdm(range(X_train.shape[0])):
    kp, des = sift.detectAndCompute(X_train[i], None)
    sift_sample = np.zeros(k)
    nkp = np.size(kp)
    if des is not None and nkp > 0:
        for d in des:
            idx = sift_kmeans.predict([d])
            sift_sample[idx] += 1 / nkp
        X_sift_train.append(sift_sample)
        y_sift_train.append(y_train[i])

X_sift_train = np.array(X_sift_train)
y_sift_train = np.array(y_sift_train)

X_sift_test = []
y_sift_test = []

for i in tqdm(range(X_test.shape[0])):
    kp, des = sift.detectAndCompute(X_test[i], None)
    sift_sample = np.zeros(k)
    nkp = np.size(kp)
    if des is not None and nkp > 0:
        for d in des:
            idx = sift_kmeans.predict([d])
            sift_sample[idx] += 1 / nkp
        X_sift_test.append(sift_sample)
        y_sift_test.append(y_test[i])

X_sift_test = np.array(X_sift_test)
y_sift_test = np.array(y_sift_test)

def model_stats(name, y_test_true, y_pred, y_pred_proba):
    cm = confusion_matrix(y_test_true, y_pred)
    print(name)
    accuracy = accuracy_score(y_test_true, y_pred)
    print("The accuracy of the model is " + str(round(accuracy, 5)))
    roc_score = roc_auc_score(y_test_true, y_pred_proba, multi_class='ovo')
    print("The ROC AUC Score of the model is " + str(round(roc_score, 5)))
    return cm

sift_mlp = MLPClassifier(random_state=101, max_iter=900000)
sift_mlp.fit(X_sift_train, y_sift_train)

y_pred = sift_mlp.predict(X_sift_test)
y_pred_proba = sift_mlp.predict_proba(X_sift_test)

sift_cm = model_stats("SIFT MLP Model", y_sift_test, y_pred, y_pred_proba)

def plot_cm(name, cm):
    classes_cm = ['akiec', 'bcc', 'bkl', 'df', 'mel', 'nv', 'vasc']
    df_cm = pd.DataFrame(cm, index=[i for i in classes_cm], columns=[i for i in classes_cm])
    df_cm = df_cm.round(5)
    plt.figure(figsize=(12, 8))
    sns.heatmap(df_cm, annot=True, fmt='g')
    plt.title(name + " Model Confusion Matrix")
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.show()

plot_cm("SIFT", sift_cm)

X_sift = np.vstack((X_sift_train, X_sift_test))
y_sift = np.append(y_sift_train, y_sift_test)

sift_tsne = TSNE(n_components=2, init="random", learning_rate=200.0)
X_sift_tsne = sift_tsne.fit_transform(X_sift)

colors_sift = ['red', 'green', 'blue', 'purple', 'black', 'brown', 'orange']
classes_sift = ['akiec', 'bcc', 'bkl', 'df', 'mel', 'nv', 'vasc']

plot_visualization(X_sift_tsne, y_sift, colors_sift, classes_sift, "T-SNE Representation")

IMGS_TO_CHECK = 10

def plot_otsu(image_index):
    lesion_img = X_g[image_index]
    _, otsu_img = cv2.threshold(lesion_img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    cv2_imshow(X[image_index])
    cv2_imshow(otsu_img)

interact(plot_otsu, image_index=widgets.IntSlider(min=0, max=IMGS_TO_CHECK - 1, step=1))

print("K Means Clustering")
kmlist = []
for i in range(IMGS_TO_CHECK):
    lesion_img = X[i]
    lesion_img_flat = lesion_img.reshape(lesion_img.shape[0] * lesion_img.shape[1],
                                         lesion_img.shape[2])
    kmeans = KMeans(n_clusters=2, random_state=101)
    kmeans_labels = kmeans.fit_predict(lesion_img_flat)
    for j in range(len(kmeans_labels)):
        if kmeans_labels[j] == 1:
            kmeans_labels[j] = 255
    kmeans_lesion_img = kmeans_labels.reshape(IMG_HEIGHT, IMG_WIDTH)
    kmlist.append(kmeans_lesion_img)

def plot_kmeans_seg(image_index):
    lesion_img = X_g[image_index]
    kmeans_lesion_img = kmlist[image_index]
    cv2_imshow(lesion_img)
    cv2_imshow(kmeans_lesion_img)

interact(plot_kmeans_seg, image_index=widgets.IntSlider(min=0, max=IMGS_TO_CHECK - 1, step=1))

lesion_img = X[0]
lesion_img_flat = lesion_img.reshape(lesion_img.shape[0] * lesion_img.shape[1],
                                     lesion_img.shape[2])

kmeans = KMeans(n_clusters=2, random_state=101)
kmeans_labels = kmeans.fit_predict(lesion_img_flat)
for i in range(len(kmeans_labels)):
    if kmeans_labels[i] == 1:
        kmeans_labels[i] = 255

tsne = TSNE(n_components=2, init='random', learning_rate=200.0)
lesion_img_tsne = tsne.fit_transform(lesion_img_flat)

colors_kmeans = ['red', 'blue']
classes_kmeans = [0, 1]

plot_visualization(lesion_img_tsne, kmeans_labels, colors_kmeans,
                   classes_kmeans, "T-SNE K Means Clustering Representation")

print("Agglomerative Clustering")
aglist = []
for i in range(IMGS_TO_CHECK):
    lesion_img = X[i]
    lesion_img_flat = lesion_img.reshape(lesion_img.shape[0] * lesion_img.shape[1],
                                         lesion_img.shape[2])
    agglomerative = AgglomerativeClustering(n_clusters=2)
    agglomerative_labels = agglomerative.fit_predict(lesion_img_flat)
    for j in range(len(agglomerative_labels)):
        if agglomerative_labels[j] == 1:
            agglomerative_labels[j] = 255
    agglomerative_lesion_img = agglomerative_labels.reshape(IMG_HEIGHT, IMG_WIDTH)
    aglist.append(agglomerative_lesion_img)

def plot_agglomerative_seg(image_index):
    lesion_img = X_g[image_index]
    agglomerative_lesion_img = aglist[image_index]
    cv2_imshow(lesion_img)
    cv2_imshow(agglomerative_lesion_img)

interact(plot_agglomerative_seg,
         image_index=widgets.IntSlider(min=0, max=IMGS_TO_CHECK - 1, step=1))

lesion_img = X[0]
lesion_img_flat = lesion_img.reshape(lesion_img.shape[0] * lesion_img.shape[1],
                                     lesion_img.shape[2])

agglomerative = AgglomerativeClustering(n_clusters=2)
agglomerative_labels = agglomerative.fit_predict(lesion_img_flat)
for i in range(len(agglomerative_labels)):
    if agglomerative_labels[i] == 1:
        agglomerative_labels[i] = 255

tsne = TSNE(n_components=2, init="random", learning_rate=200.0)
lesion_img_tsne = tsne.fit_transform(lesion_img_flat)

colors_ag = ['red', 'blue']
classes_ag = [0, 1]

plot_visualization(lesion_img_tsne, agglomerative_labels,
                   colors_ag, classes_ag,
                   "T-SNE Agglomerative Clustering Representation")

os.makedirs('images_seg', exist_ok=True)
!wget -O images_seg.zip 'https://storage.googleapis.com/inspirit-ai-data-bucket-1/Data/AI%20Scholars/Sessions%206%20-%2010%20(Projects)/Project%20-%20(Healthcare%20B)%20Skin%20Cancer%20Diagnosis/images_seg.zip'
!unzip -q images_seg.zip -d images_seg

os.makedirs('segmentation', exist_ok=True)
!wget -O segmentations.zip 'https://storage.googleapis.com/inspirit-ai-data-bucket-1/Data/AI%20Scholars/Sessions%206%20-%2010%20(Projects)/Project%20-%20(Healthcare%20B)%20Skin%20Cancer%20Diagnosis/segmentations.zip'
!unzip -q segmentations.zip -d segmentation

IMG_SEG_WIDTH = 256
IMG_SEG_HEIGHT = 192

X_seg = []
y_seg = []

seg_path, seg_dirs, files = next(os.walk("images_seg/Images"))
seg_path, seg_dirs, files_seg = next(os.walk("segmentation/Segmentation"))

for i in tqdm(range(len(files))):
    file_name = files[i].split('.')[0]
    seg_index = [j for j, s in enumerate(files_seg) if file_name in s][0]

    img = cv2.imread('images_seg/Images/' + files[i], cv2.IMREAD_COLOR)
    img = cv2.resize(img, (IMG_SEG_WIDTH, IMG_SEG_HEIGHT))
    img = img / 255.0
    X_seg.append(img)

    img = cv2.imread('segmentation/Segmentation/' + files_seg[seg_index],
                     cv2.IMREAD_COLOR)
    img = cv2.resize(img, (IMG_SEG_WIDTH, IMG_SEG_HEIGHT))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = img / 255.0
    y_seg.append(img)

X_seg = np.array(X_seg)
y_seg = np.array(y_seg)

print("Original Image")
cv2_imshow(X_seg[2] * 255)
print("Image Mask")
cv2_imshow(y_seg[2] * 255)

X_seg_train, X_seg_test, y_seg_train, y_seg_test = train_test_split(
    X_seg, y_seg, test_size=0.2, random_state=101
)

def build_model():
    inputs = Input((IMG_SEG_HEIGHT, IMG_SEG_WIDTH, 3))
    s = Lambda(lambda x: x / 255)(inputs)

    conv_blocks = [16, 32, 64, 128, 256, 128, 64, 32, 16]

    conv1 = Conv2D(conv_blocks[0], (3, 3), activation='elu',
                   kernel_initializer='he_normal', padding='same')(s)
    conv1 = Dropout(0.1)(conv1)
    conv1 = Conv2D(conv_blocks[0], (3, 3), activation='elu',
                   kernel_initializer='he_normal', padding='same')(conv1)
    pool1 = MaxPooling2D((2, 2))(conv1)

    conv2 = Conv2D(conv_blocks[1], (3, 3), activation='elu',
                   kernel_initializer='he_normal', padding='same')(pool1)
    conv2 = Dropout(0.1)(conv2)
    conv2 = Conv2D(conv_blocks[1], (3, 3), activation='elu',
                   kernel_initializer='he_normal', padding='same')(conv2)
    pool2 = MaxPooling2D((2, 2))(conv2)

    conv3 = Conv2D(conv_blocks[2], (3, 3), activation='elu',
                   kernel_initializer='he_normal', padding='same')(pool2)
    conv3 = Dropout(0.2)(conv3)
    conv3 = Conv2D(conv_blocks[2], (3, 3), activation='elu',
                   kernel_initializer='he_normal', padding='same')(conv3)
    pool3 = MaxPooling2D((2, 2))(conv3)

    conv4 = Conv2D(conv_blocks[3], (3, 3), activation='elu',
                   kernel_initializer='he_normal', padding='same')(pool3)
    conv4 = Dropout(0.2)(conv4)
    conv4 = Conv2D(conv_blocks[3], (3, 3), activation='elu',
                   kernel_initializer='he_normal', padding='same')(conv4)
    pool4 = MaxPooling2D(pool_size=(2, 2))(conv4)

    conv5 = Conv2D(conv_blocks[4], (3, 3), activation='elu',
                   kernel_initializer='he_normal', padding='same')(pool4)
    conv5 = Dropout(0.3)(conv5)
    conv5 = Conv2D(conv_blocks[4], (3, 3), activation='elu',
                   kernel_initializer='he_normal', padding='same')(conv5)

    upconv6 = Conv2DTranspose(conv_blocks[5], (2, 2),
                              strides=(2, 2), padding='same')(conv5)
    upconv6 = concatenate([upconv6, conv4])
    conv6 = Conv2D(conv_blocks[5], (3, 3), activation='elu',
                   kernel_initializer='he_normal', padding='same')(upconv6)
    conv6 = Dropout(0.2)(conv6)
    conv6 = Conv2D(conv_blocks[5], (3, 3), activation='elu',
                   kernel_initializer='he_normal', padding='same')(conv6)

    upconv7 = Conv2DTranspose(conv_blocks[6], (2, 2),
                              strides=(2, 2), padding='same')(conv6)
    upconv7 = concatenate([upconv7, conv3])
    conv7 = Conv2D(conv_blocks[6], (3, 3), activation='elu',
                   kernel_initializer='he_normal', padding='same')(upconv7)
    conv7 = Dropout(0.2)(conv7)
    conv7 = Conv2D(conv_blocks[6], (3, 3), activation='elu',
                   kernel_initializer='he_normal', padding='same')(conv7)

    upconv8 = Conv2DTranspose(conv_blocks[7], (2, 2),
                              strides=(2, 2), padding='same')(conv7)
    upconv8 = concatenate([upconv8, conv2])
    conv8 = Conv2D(conv_blocks[7], (3, 3), activation='elu',
                   kernel_initializer='he_normal', padding='same')(upconv8)
    conv8 = Dropout(0.1)(conv8)
    conv8 = Conv2D(conv_blocks[7], (3, 3), activation='elu',
                   kernel_initializer='he_normal', padding='same')(conv8)

    upconv9 = Conv2DTranspose(conv_blocks[8], (2, 2),
                              strides=(2, 2), padding='same')(conv8)
    upconv9 = concatenate([upconv9, conv1], axis=3)
    conv9 = Conv2D(conv_blocks[8], (3, 3), activation='elu',
                   kernel_initializer='he_normal', padding='same')(upconv9)
    conv9 = Dropout(0.1)(conv9)
    conv9 = Conv2D(conv_blocks[8], (3, 3), activation='elu',
                   kernel_initializer='he_normal', padding='same')(conv9)

    outputs = Conv2D(1, (1, 1), activation='sigmoid')(conv9)
    model = Model(inputs=[inputs], outputs=[outputs])
    return model

def iou(y_true, y_pred):
    def f(y_true_np, y_pred_np):
        intersection = (y_true_np * y_pred_np).sum()
        union = y_true_np.sum() + y_pred_np.sum() - intersection
        x = (intersection + 1e-15) / (union + 1e-15)
        x = x.astype(np.float32)
        return x
    return tf.numpy_function(f, [y_true, y_pred], tf.float32)

batch = 16
lr = 1e-4

model = build_model()
opt = keras.optimizers.Adam(lr)
metrics = [iou]
model.compile(loss="binary_crossentropy", optimizer=opt, metrics=metrics)

model.fit(X_seg_train.astype(np.float32), y_seg_train.astype(np.float32),
          validation_data=(X_seg_test.astype(np.float32), y_seg_test.astype(np.float32)),
          epochs=20, verbose=1)

y_seg_pred = model.predict(X_seg_test)
iou_val = iou(np.expand_dims(y_seg_test, axis=3), y_seg_pred)
print("IOU: " + str(iou_val.numpy()))

image_index = 27

print("Original Image")
cv2_imshow(X_seg_test[image_index] * 255)

print("True Segmentation")
cv2_imshow(y_seg_test[image_index] * 255)

print("Predicted Segmentation")
prediction = model.predict(X_seg_test[image_index][None, :])
prediction_img = np.reshape(prediction.flatten(), (IMG_SEG_HEIGHT, IMG_SEG_WIDTH))
_, threshold = cv2.threshold(prediction_img, 0.5, 255, cv2.THRESH_BINARY)
cv2_imshow(threshold * 255)

def build_model_modified():
    inputs = Input((IMG_SEG_HEIGHT, IMG_SEG_WIDTH, 3))
    s = Lambda(lambda x: x / 255)(inputs)

    conv_blocks = [16, 32, 64, 128, 256, 128, 64, 32, 16]

    conv1 = Conv2D(conv_blocks[0], (3, 3), activation='elu',
                   kernel_initializer='he_normal', padding='same')(s)
    conv1 = Dropout(0.1)(conv1)
    conv1 = Conv2D(conv_blocks[0], (3, 3), activation='elu',
                   kernel_initializer='he_normal', padding='same')(conv1)
    pool1 = MaxPooling2D((2, 2))(conv1)

    conv2 = Conv2D(conv_blocks[1], (3, 3), activation='elu',
                   kernel_initializer='he_normal', padding='same')(pool1)
    conv2 = Dropout(0.1)(conv2)
    conv2 = Conv2D(conv_blocks[1], (3, 3), activation='elu',
                   kernel_initializer='he_normal', padding='same')(conv2)
    pool2 = MaxPooling2D((2, 2))(conv2)

    conv3 = Conv2D(conv_blocks[2], (3, 3), activation='elu',
                   kernel_initializer='he_normal', padding='same')(pool2)
    conv3 = Dropout(0.2)(conv3)
    conv3 = Conv2D(conv_blocks[2], (3, 3), activation='elu',
                   kernel_initializer='he_normal', padding='same')(conv3)
    pool3 = MaxPooling2D((2, 2))(conv3)

    conv4 = Conv2D(conv_blocks[3], (3, 3), activation='elu',
                   kernel_initializer='he_normal', padding='same')(pool3)
    conv4 = Dropout(0.2)(conv4)
    conv4 = Conv2D(conv_blocks[3], (3, 3), activation='elu',
                   kernel_initializer='he_normal', padding='same')(conv4)
    pool4 = MaxPooling2D(pool_size=(2, 2))(conv4)

    conv5 = Conv2D(conv_blocks[4], (3, 3), activation='elu',
                   kernel_initializer='he_normal', padding='same')(pool4)
    conv5 = Dropout(0.3)(conv5)
    conv5 = Conv2D(conv_blocks[4], (3, 3), activation='elu',
                   kernel_initializer='he_normal', padding='same')(conv5)

    upconv6 = Conv2DTranspose(conv_blocks[5], (2, 2),
                              strides=(2, 2), padding='same')(conv5)
    upconv6 = concatenate([upconv6, conv4])
    conv6 = Conv2D(conv_blocks[5], (3, 3), activation='elu',
                   kernel_initializer='he_normal', padding='same')(upconv6)
    conv6 = Dropout(0.2)(conv6)
    conv6 = Conv2D(conv_blocks[5], (3, 3), activation='elu',
                   kernel_initializer='he_normal', padding='same')(conv6)

    upconv7 = Conv2DTranspose(conv_blocks[6], (2, 2),
                              strides=(2, 2), padding='same')(conv6)
    upconv7 = concatenate([upconv7, conv3])
    conv7 = Conv2D(conv_blocks[6], (3, 3), activation='elu',
                   kernel_initializer='he_normal', padding='same')(upconv7)
    conv7 = Dropout(0.2)(conv7)
    conv7 = Conv2D(conv_blocks[6], (3, 3), activation='elu',
                   kernel_initializer='he_normal', padding='same')(conv7)

    upconv8 = Conv2DTranspose(conv_blocks[7], (2, 2),
                              strides=(2, 2), padding='same')(conv7)
    upconv8 = concatenate([upconv8, conv2])
    conv8 = Conv2D(conv_blocks[7], (3, 3), activation='elu',
                   kernel_initializer='he_normal', padding='same')(upconv8)
    conv8 = Dropout(0.1)(conv8)
    conv8 = Conv2D(conv_blocks[7], (3, 3), activation='elu',
                   kernel_initializer='he_normal', padding='same')(conv8)

    upconv9 = Conv2DTranspose(conv_blocks[8], (2, 2),
                              strides=(2, 2), padding='same')(conv8)
    upconv9 = concatenate([upconv9, conv1], axis=3)
    conv9 = Conv2D(conv_blocks[8], (3, 3), activation='elu',
                   kernel_initializer='he_normal', padding='same')(upconv9)
    conv9 = Dropout(0.1)(conv9)
    c9 = Conv2D(conv_blocks[8], (3, 3), activation='elu',
                kernel_initializer='he_normal', padding='same')(conv9)

    outputs = Conv2D(1, (1, 1), activation='sigmoid')(c9)
    model_mod = Model(inputs=[inputs], outputs=[outputs])
    return model_mod

batch = 16
lr = 1e-4

model_mod = build_model_modified()
opt_mod = keras.optimizers.Adam(lr)
metrics_mod = [iou]
model_mod.compile(loss="binary_crossentropy", optimizer=opt_mod, metrics=metrics_mod)

model_mod.fit(X_seg_train.astype(np.float32), y_seg_train.astype(np.float32),
              validation_data=(X_seg_test.astype(np.float32), y_seg_test.astype(np.float32)),
              epochs=20, verbose=1)

y_seg_pred_mod = model_mod.predict(X_seg_test)
iou_val_mod = iou(np.expand_dims(y_seg_test, axis=3), y_seg_pred_mod)
print("Modified U-Net IOU: " + str(iou_val_mod.numpy()))

image_index = 45

print("Original Image")
cv2_imshow(X_seg_test[image_index] * 255)

print("True Segmentation")
cv2_imshow(y_seg_test[image_index] * 255)

print("Predicted Segmentation (Modified U-Net)")
prediction_mod = model_mod.predict(X_seg_test[image_index][None, :])
prediction_img_mod = np.reshape(prediction_mod.flatten(), (IMG_SEG_HEIGHT, IMG_SEG_WIDTH))
_, threshold_mod = cv2.threshold(prediction_img_mod, 0.5, 255, cv2.THRESH_BINARY)
cv2_imshow(threshold_mod * 255)
