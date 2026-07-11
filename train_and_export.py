# ============================================================
# SkIntel — self-contained U-Net segmentation training + export
# Paste this into a SINGLE Google Colab cell (Runtime = T4 GPU) and run.
# It downloads the data, trains the U-Net, and downloads skintel_unet.keras.
# Modern imports — does NOT depend on the notebook's old/broken cells.
# ============================================================

import os
import cv2
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import (
    Input, Conv2D, Dropout, MaxPooling2D, Conv2DTranspose, concatenate, Lambda
)
from tensorflow.keras.models import Model
from sklearn.model_selection import train_test_split

IMG_SEG_WIDTH = 256
IMG_SEG_HEIGHT = 192

# --- 1. Download the segmentation dataset -------------------
BASE = ('https://storage.googleapis.com/inspirit-ai-data-bucket-1/Data/'
        'AI%20Scholars/Sessions%206%20-%2010%20(Projects)/'
        'Project%20-%20(Healthcare%20B)%20Skin%20Cancer%20Diagnosis')

if not os.path.exists('images_seg'):
    !wget -q -O images_seg.zip '{BASE}/images_seg.zip'
    !unzip -q -o images_seg.zip -d images_seg
if not os.path.exists('segmentation'):
    !wget -q -O segmentations.zip '{BASE}/segmentations.zip'
    !unzip -q -o segmentations.zip -d segmentation

# --- 2. Build X_seg / y_seg (same preprocessing as the notebook) ---
img_dir = 'images_seg/Images'
seg_dir = 'segmentation/Segmentation'
_, _, img_files = next(os.walk(img_dir))
_, _, seg_files = next(os.walk(seg_dir))

X_seg, y_seg = [], []
for f in img_files:
    name = f.split('.')[0]
    matches = [s for s in seg_files if name in s]
    if not matches:
        continue
    img = cv2.imread(os.path.join(img_dir, f), cv2.IMREAD_COLOR)   # BGR
    img = cv2.resize(img, (IMG_SEG_WIDTH, IMG_SEG_HEIGHT)) / 255.0
    X_seg.append(img)

    seg = cv2.imread(os.path.join(seg_dir, matches[0]), cv2.IMREAD_COLOR)
    seg = cv2.resize(seg, (IMG_SEG_WIDTH, IMG_SEG_HEIGHT))
    seg = cv2.cvtColor(seg, cv2.COLOR_BGR2GRAY) / 255.0
    y_seg.append(seg)

X_seg = np.array(X_seg, dtype=np.float32)
y_seg = np.array(y_seg, dtype=np.float32)
print('Data:', X_seg.shape, y_seg.shape)

X_tr, X_te, y_tr, y_te = train_test_split(
    X_seg, y_seg, test_size=0.2, random_state=101
)

# --- 3. U-Net (identical architecture to the notebook) ------
def build_model():
    inputs = Input((IMG_SEG_HEIGHT, IMG_SEG_WIDTH, 3))
    s = Lambda(lambda x: x / 255)(inputs)
    cb = [16, 32, 64, 128, 256, 128, 64, 32, 16]

    c1 = Conv2D(cb[0], 3, activation='elu', kernel_initializer='he_normal', padding='same')(s)
    c1 = Dropout(0.1)(c1)
    c1 = Conv2D(cb[0], 3, activation='elu', kernel_initializer='he_normal', padding='same')(c1)
    p1 = MaxPooling2D((2, 2))(c1)

    c2 = Conv2D(cb[1], 3, activation='elu', kernel_initializer='he_normal', padding='same')(p1)
    c2 = Dropout(0.1)(c2)
    c2 = Conv2D(cb[1], 3, activation='elu', kernel_initializer='he_normal', padding='same')(c2)
    p2 = MaxPooling2D((2, 2))(c2)

    c3 = Conv2D(cb[2], 3, activation='elu', kernel_initializer='he_normal', padding='same')(p2)
    c3 = Dropout(0.2)(c3)
    c3 = Conv2D(cb[2], 3, activation='elu', kernel_initializer='he_normal', padding='same')(c3)
    p3 = MaxPooling2D((2, 2))(c3)

    c4 = Conv2D(cb[3], 3, activation='elu', kernel_initializer='he_normal', padding='same')(p3)
    c4 = Dropout(0.2)(c4)
    c4 = Conv2D(cb[3], 3, activation='elu', kernel_initializer='he_normal', padding='same')(c4)
    p4 = MaxPooling2D((2, 2))(c4)

    c5 = Conv2D(cb[4], 3, activation='elu', kernel_initializer='he_normal', padding='same')(p4)
    c5 = Dropout(0.3)(c5)
    c5 = Conv2D(cb[4], 3, activation='elu', kernel_initializer='he_normal', padding='same')(c5)

    u6 = Conv2DTranspose(cb[5], 2, strides=(2, 2), padding='same')(c5)
    u6 = concatenate([u6, c4])
    c6 = Conv2D(cb[5], 3, activation='elu', kernel_initializer='he_normal', padding='same')(u6)
    c6 = Dropout(0.2)(c6)
    c6 = Conv2D(cb[5], 3, activation='elu', kernel_initializer='he_normal', padding='same')(c6)

    u7 = Conv2DTranspose(cb[6], 2, strides=(2, 2), padding='same')(c6)
    u7 = concatenate([u7, c3])
    c7 = Conv2D(cb[6], 3, activation='elu', kernel_initializer='he_normal', padding='same')(u7)
    c7 = Dropout(0.2)(c7)
    c7 = Conv2D(cb[6], 3, activation='elu', kernel_initializer='he_normal', padding='same')(c7)

    u8 = Conv2DTranspose(cb[7], 2, strides=(2, 2), padding='same')(c7)
    u8 = concatenate([u8, c2])
    c8 = Conv2D(cb[7], 3, activation='elu', kernel_initializer='he_normal', padding='same')(u8)
    c8 = Dropout(0.1)(c8)
    c8 = Conv2D(cb[7], 3, activation='elu', kernel_initializer='he_normal', padding='same')(c8)

    u9 = Conv2DTranspose(cb[8], 2, strides=(2, 2), padding='same')(c8)
    u9 = concatenate([u9, c1], axis=3)
    c9 = Conv2D(cb[8], 3, activation='elu', kernel_initializer='he_normal', padding='same')(u9)
    c9 = Dropout(0.1)(c9)
    c9 = Conv2D(cb[8], 3, activation='elu', kernel_initializer='he_normal', padding='same')(c9)

    outputs = Conv2D(1, 1, activation='sigmoid')(c9)
    return Model(inputs=[inputs], outputs=[outputs])

# --- 4. Train ----------------------------------------------
model = build_model()
model.compile(loss='binary_crossentropy',
              optimizer=keras.optimizers.Adam(learning_rate=1e-4))
model.fit(X_tr, y_tr, validation_data=(X_te, y_te), epochs=20, verbose=1)

# --- 5. Save + download ------------------------------------
model.save('skintel_unet.keras')
print('TensorFlow version:', tf.__version__)   # <-- note this for requirements.txt
from google.colab import files
files.download('skintel_unet.keras')
