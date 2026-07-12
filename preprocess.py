import cv2
import numpy as np

# Segmentation input size (must match the U-Net training constants).
SEG_WIDTH = 256
SEG_HEIGHT = 192

# Classifier input size (matches the 75x100 HAM10000 subset arrays).
CLS_WIDTH = 100
CLS_HEIGHT = 75


def _to_net_input(image, width, height):
    """RGB uint8 image -> normalized BGR tensor with a batch dim.

    Gradio hands us RGB. Both models were trained on cv2-read BGR data that was
    never converted, so we convert here and divide by 255 to match.
    """
    bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    resized = cv2.resize(bgr, (width, height))  # cv2 takes (width, height)
    return (resized / 255.0).astype(np.float32)[None, ...]


def preprocess_segmentation(image):
    return _to_net_input(image, SEG_WIDTH, SEG_HEIGHT)


def preprocess_classifier(image):
    return _to_net_input(image, CLS_WIDTH, CLS_HEIGHT)
