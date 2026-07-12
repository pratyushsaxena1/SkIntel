# ============================================================
# SkIntel: binary malignancy classifier (train + save).
# Paste into a SINGLE Google Colab cell (Runtime = T4 GPU) and run.
# Downloads the balanced HAM10000 subset, trains a small CNN to predict
# malignant vs benign, reports held-out metrics, and downloads
# skintel_classifier.keras.
# ============================================================

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight

IMG_H, IMG_W = 75, 100

BASE = ('https://storage.googleapis.com/inspirit-ai-data-bucket-1/Data/'
        'AI%20Scholars/Sessions%206%20-%2010%20(Projects)/'
        'Project%20-%20(Healthcare%20B)%20Skin%20Cancer%20Diagnosis')

# --- 1. Load the preprocessed subset (BGR uint8, 7-class labels) ---
!wget -q -O X.npy '{BASE}/X.npy'
!wget -q -O y.npy '{BASE}/y.npy'
X = np.load('X.npy')     # (967, 75, 100, 3) uint8, BGR
y7 = np.load('y.npy')    # (967,) int64, classes 0..6

# --- 2. Collapse 7 classes to binary malignant/benign ---
# akiec(0), bcc(1), mel(4) are malignant; bkl(2), df(3), nv(5), vasc(6) benign.
MALIGNANT = [0, 1, 4]
y = np.isin(y7, MALIGNANT).astype(np.float32)
print('malignant:', int(y.sum()), 'benign:', int((1 - y).sum()))

# --- 3. Normalize and split (no rescale layer inside the model) ---
X = X.astype(np.float32) / 255.0
X_tr, X_te, y_tr, y_te = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=101
)

# --- 4. Small CNN. Pure inference graph: no augmentation layers inside it, so
# it converts to ONNX cleanly (in-graph RandomFlip/RandomRotation emit ops
# tf2onnx cannot export). Augmentation lives in the input pipeline below. ---
def build_model():
    inputs = keras.Input((IMG_H, IMG_W, 3))
    x = inputs
    for f in (32, 64, 128):
        x = layers.Conv2D(f, 3, padding='same', activation='relu')(x)
        x = layers.Conv2D(f, 3, padding='same', activation='relu')(x)
        x = layers.MaxPooling2D()(x)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(0.3)(x)
    x = layers.Dense(64, activation='relu')(x)
    outputs = layers.Dense(1, activation='sigmoid')(x)
    return keras.Model(inputs, outputs)

model = build_model()
model.compile(
    optimizer=keras.optimizers.Adam(1e-4),
    loss='binary_crossentropy',
    metrics=['accuracy', keras.metrics.AUC(name='auc')],
)

# --- 5. Class weights favor the smaller malignant class ---
cw = compute_class_weight('balanced', classes=np.array([0.0, 1.0]), y=y_tr)
class_weight = {0: float(cw[0]), 1: float(cw[1])}

# Augmentation is applied only to the training data through tf.data, keeping it
# out of the saved model. Flip and small rotation add light regularization.
augment = keras.Sequential([
    layers.RandomFlip('horizontal'),
    layers.RandomRotation(0.05),
])
AUTOTUNE = tf.data.AUTOTUNE
train_ds = (
    tf.data.Dataset.from_tensor_slices((X_tr, y_tr))
    .shuffle(1024)
    .batch(32)
    .map(lambda a, b: (augment(a, training=True), b), num_parallel_calls=AUTOTUNE)
    .prefetch(AUTOTUNE)
)
val_ds = tf.data.Dataset.from_tensor_slices((X_te, y_te)).batch(32).prefetch(AUTOTUNE)

model.fit(train_ds, validation_data=val_ds, epochs=40,
          class_weight=class_weight, verbose=1)

# --- 6. Held-out metrics: accuracy, AUC, sensitivity, specificity ---
p = model.predict(X_te, verbose=0).ravel()
pred = (p > 0.5).astype(np.float32)
tp = int(((pred == 1) & (y_te == 1)).sum())
tn = int(((pred == 0) & (y_te == 0)).sum())
fp = int(((pred == 1) & (y_te == 0)).sum())
fn = int(((pred == 0) & (y_te == 1)).sum())
sensitivity = tp / (tp + fn) if (tp + fn) else 0.0
specificity = tn / (tn + fp) if (tn + fp) else 0.0
auc = keras.metrics.AUC()(y_te, p).numpy()
print(f'accuracy={(tp + tn) / len(y_te):.3f} auc={auc:.3f} '
      f'sensitivity={sensitivity:.3f} specificity={specificity:.3f}')

model.save('skintel_classifier.keras')
from google.colab import files
files.download('skintel_classifier.keras')
