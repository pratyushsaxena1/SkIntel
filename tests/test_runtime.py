"""Runtime checks with synthetic inputs, independent of the training dataset."""
import numpy as np

from inference import analyze


def test_empty_upload():
    assert analyze(None) == (None, None, None)


def test_bundled_models_accept_rgb_input():
    image = np.random.default_rng(101).integers(0, 256, (120, 90, 3), dtype=np.uint8)
    overlay, mask, scores = analyze(image)
    assert overlay.shape == (192, 256, 3)
    assert overlay.dtype == np.uint8
    assert mask.shape == (192, 256)
    assert set(np.unique(mask)) <= {0, 255}
    assert set(scores) == {"Benign", "Possibly malignant"}
    assert all(0 <= value <= 1 for value in scores.values())
    assert abs(sum(scores.values()) - 1) < 1e-6
