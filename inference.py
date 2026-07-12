import cv2
import numpy as np
import onnxruntime as ort

from preprocess import preprocess_segmentation, preprocess_classifier

# Load both ONNX models once at import (portable, no TensorFlow needed).
seg_session = ort.InferenceSession(
    "skintel_unet.onnx", providers=["CPUExecutionProvider"]
)
SEG_INPUT = seg_session.get_inputs()[0].name
SEG_OUTPUT = seg_session.get_outputs()[0].name

cls_session = ort.InferenceSession(
    "skintel_classifier.onnx", providers=["CPUExecutionProvider"]
)
CLS_INPUT = cls_session.get_inputs()[0].name
CLS_OUTPUT = cls_session.get_outputs()[0].name


def segment(image):
    """Return (mask overlay, binary mask) for an RGB image."""
    net_input = preprocess_segmentation(image)
    pred = seg_session.run([SEG_OUTPUT], {SEG_INPUT: net_input})[0]
    mask = (pred[0, :, :, 0] > 0.5).astype(np.uint8)

    # Rebuild the (BGR) model-resolution image from the tensor for display.
    resized = (net_input[0] * 255.0).astype(np.uint8)
    display_rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
    overlay = display_rgb.copy()
    overlay[mask == 1] = (255, 0, 0)
    blended = cv2.addWeighted(display_rgb, 0.6, overlay, 0.4, 0)

    binary_mask = (mask * 255).astype(np.uint8)
    return blended, binary_mask


def classify(image):
    """Return P(malignant) in [0, 1] for an RGB image."""
    net_input = preprocess_classifier(image)
    pred = cls_session.run([CLS_OUTPUT], {CLS_INPUT: net_input})[0]
    return float(pred.ravel()[0])


def analyze(image):
    """Segment the lesion and estimate malignancy from one image."""
    if image is None:
        return None, None, None
    overlay, binary_mask = segment(image)
    p = classify(image)
    return overlay, binary_mask, {"Benign": 1.0 - p, "Possibly malignant": p}
