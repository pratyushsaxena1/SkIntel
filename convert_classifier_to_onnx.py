# ============================================================
# Convert the trained Keras classifier -> ONNX (run in Google Colab).
# Paste into a Colab cell in the SAME notebook where
# skintel_classifier.keras exists (or after re-uploading it).
# Downloads skintel_classifier.onnx.
# ============================================================

!pip install -q tf2onnx onnx onnxruntime

import tensorflow as tf

model = tf.keras.models.load_model(
    "skintel_classifier.keras", compile=False, safe_mode=False
)

# SavedModel first, then tf2onnx (robust, no Python-version fragility).
model.export("skintel_classifier_saved_model")

!python -m tf2onnx.convert \
    --saved-model skintel_classifier_saved_model \
    --output skintel_classifier.onnx \
    --opset 13

# Parity check: Keras vs ONNX on random inputs must match within tolerance.
import numpy as np
import onnxruntime as ort

sample = np.random.rand(4, 75, 100, 3).astype(np.float32)
keras_out = model.predict(sample, verbose=0).ravel()

sess = ort.InferenceSession(
    "skintel_classifier.onnx", providers=["CPUExecutionProvider"]
)
name_in = sess.get_inputs()[0].name
name_out = sess.get_outputs()[0].name
onnx_out = sess.run([name_out], {name_in: sample})[0].ravel()

print("INPUT :", [(i.name, i.shape) for i in sess.get_inputs()])
print("OUTPUT:", [(o.name, o.shape) for o in sess.get_outputs()])
print("max abs diff:", float(np.max(np.abs(keras_out - onnx_out))))

from google.colab import files
files.download("skintel_classifier.onnx")
