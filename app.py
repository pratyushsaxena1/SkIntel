# --- Work around a gradio_client bug: get_type() does `"const" in schema`,
# which crashes when a JSON schema value is a bool (e.g. additionalProperties:
# true). Patch the schema walkers to tolerate bool schemas. Must run before
# gradio builds any API info.
import gradio_client.utils as _gcu

_orig_json_to_type = _gcu._json_schema_to_python_type


def _safe_json_to_type(schema, defs=None):
    if isinstance(schema, bool):
        return "Any"
    return _orig_json_to_type(schema, defs)


_gcu._json_schema_to_python_type = _safe_json_to_type

_orig_get_type = _gcu.get_type


def _safe_get_type(schema):
    if not isinstance(schema, dict):
        return "Any"
    return _orig_get_type(schema)


_gcu.get_type = _safe_get_type
# --- end workaround ---

import os
import cv2
import numpy as np
import gradio as gr
import onnxruntime as ort

# Must match the training constants in the notebook:
#   IMG_SEG_WIDTH = 256, IMG_SEG_HEIGHT = 192
IMG_SEG_WIDTH = 256
IMG_SEG_HEIGHT = 192

# Load the ONNX model once at startup (portable — no TensorFlow needed).
session = ort.InferenceSession(
    "skintel_unet.onnx", providers=["CPUExecutionProvider"]
)
INPUT_NAME = session.get_inputs()[0].name
OUTPUT_NAME = session.get_outputs()[0].name


def segment(image):
    """Take an uploaded RGB image, return (mask overlay, binary mask)."""
    if image is None:
        return None, None

    # Gradio hands us RGB. Training used cv2.imread (BGR) and never converted,
    # so the model was trained on BGR -> convert to match.
    bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    # Resize to the model's input size. cv2 takes (width, height).
    resized = cv2.resize(bgr, (IMG_SEG_WIDTH, IMG_SEG_HEIGHT))

    # Training divided by 255 AND the model has an internal Lambda(x/255)
    # (now baked into the ONNX graph), so feed 0-1 values to reproduce that.
    net_input = (resized / 255.0).astype(np.float32)[None, ...]

    # Predict -> sigmoid map in [0, 1], then threshold at 0.5.
    pred = session.run([OUTPUT_NAME], {INPUT_NAME: net_input})[0]
    mask = (pred[0, :, :, 0] > 0.5).astype(np.uint8)

    # Build a red overlay on top of the (RGB) resized image for display.
    display_rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
    overlay = display_rgb.copy()
    overlay[mask == 1] = (255, 0, 0)
    blended = cv2.addWeighted(display_rgb, 0.6, overlay, 0.4, 0)

    binary_mask = (mask * 255).astype(np.uint8)
    return blended, binary_mask


demo = gr.Interface(
    fn=segment,
    inputs=gr.Image(type="numpy", label="Skin lesion image"),
    outputs=[
        gr.Image(label="Lesion overlay"),
        gr.Image(label="Segmentation mask"),
    ],
    title="SkIntel — Skin Lesion Segmentation",
    description="Upload a dermoscopic image to outline the lesion region with a U-Net model.",
    flagging_mode="never",
)

if __name__ == "__main__":
    # Render provides the port via $PORT; default to 7860 for local runs.
    port = int(os.environ.get("PORT", 7860))
    demo.launch(
        server_name="0.0.0.0", server_port=port, ssr_mode=False, show_api=False
    )
