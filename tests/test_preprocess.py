import numpy as np

from preprocess import (
    preprocess_classifier,
    preprocess_segmentation,
    CLS_WIDTH,
    CLS_HEIGHT,
    SEG_WIDTH,
    SEG_HEIGHT,
)


def _red_rgb(h=120, w=90):
    # Pure red in RGB, so we can tell whether the channel order was swapped.
    img = np.zeros((h, w, 3), dtype=np.uint8)
    img[..., 0] = 255
    return img


def test_classifier_shape_and_range():
    out = preprocess_classifier(_red_rgb())
    assert out.shape == (1, CLS_HEIGHT, CLS_WIDTH, 3)
    assert out.dtype == np.float32
    assert out.min() >= 0.0 and out.max() <= 1.0


def test_classifier_swaps_rgb_to_bgr():
    out = preprocess_classifier(_red_rgb())[0]
    # Red must land in channel 2 (BGR), not channel 0.
    assert out[..., 2].mean() > 0.9
    assert out[..., 0].mean() < 0.1


def test_segmentation_shape_and_bgr():
    out = preprocess_segmentation(_red_rgb())
    assert out.shape == (1, SEG_HEIGHT, SEG_WIDTH, 3)
    assert out.dtype == np.float32
    assert out[0, ..., 2].mean() > 0.9
    assert out[0, ..., 0].mean() < 0.1
