# ============================================================
# Convert the trained Keras U-Net -> ONNX (run in Google Colab).
# Paste into a Colab cell in the SAME notebook where skintel_unet.keras exists
# (or after re-uploading that file). Downloads skintel_unet.onnx.
# ============================================================

!pip install -q tf2onnx onnx

import tensorflow as tf

# Load the model you already trained/saved.
model = tf.keras.models.load_model(
    "skintel_unet.keras", compile=False, safe_mode=False
)

# Export to a TF SavedModel first (robust path for tf2onnx; bakes the
# internal Lambda(x/255) into the graph, no Python-version fragility).
model.export("skintel_saved_model")

# SavedModel -> ONNX
!python -m tf2onnx.convert \
    --saved-model skintel_saved_model \
    --output skintel_unet.onnx \
    --opset 13

# Sanity check: print the input/output names + shapes.
import onnxruntime as ort
sess = ort.InferenceSession("skintel_unet.onnx", providers=["CPUExecutionProvider"])
print("INPUT :", [(i.name, i.shape) for i in sess.get_inputs()])
print("OUTPUT:", [(o.name, o.shape) for o in sess.get_outputs()])

from google.colab import files
files.download("skintel_unet.onnx")
