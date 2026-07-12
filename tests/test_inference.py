import os

import numpy as np
import pytest

# The balanced HAM10000 subset arrays (BGR uint8 images, 7-class labels).
# Set SKINTEL_DATA_DIR to the folder holding X.npy and y.npy to run these end to
# end; without it the inference tests skip so the rest of the suite still runs.
DATA_DIR = os.environ.get("SKINTEL_DATA_DIR", "")
X_PATH = os.path.join(DATA_DIR, "X.npy")
Y_PATH = os.path.join(DATA_DIR, "y.npy")

pytestmark = pytest.mark.skipif(
    not (DATA_DIR and os.path.exists(X_PATH) and os.path.exists(Y_PATH)),
    reason="set SKINTEL_DATA_DIR (with X.npy and y.npy) to run inference tests",
)

# Malignant classes: akiec(0), bcc(1), mel(4).
MALIGNANT = [0, 1, 4]


def _to_rgb(bgr):
    # The stored arrays are BGR; Gradio hands the app RGB, so flip to match.
    return bgr[..., ::-1].copy()


def test_analyze_output_contract():
    import inference

    X = np.load(X_PATH)
    overlay, mask, label = inference.analyze(_to_rgb(X[0]))
    assert overlay.shape == (192, 256, 3)
    assert mask.shape == (192, 256)
    assert set(label) == {"Benign", "Possibly malignant"}
    assert abs(sum(label.values()) - 1.0) < 1e-6
    assert all(0.0 <= v <= 1.0 for v in label.values())


def test_analyze_none_returns_empty():
    import inference

    assert inference.analyze(None) == (None, None, None)


def test_malignant_scored_higher_than_benign():
    import inference

    X = np.load(X_PATH)
    y = np.load(Y_PATH)
    mal = np.where(np.isin(y, MALIGNANT))[0][:30]
    ben = np.where(~np.isin(y, MALIGNANT))[0][:30]
    mal_p = np.mean([inference.classify(_to_rgb(X[i])) for i in mal])
    ben_p = np.mean([inference.classify(_to_rgb(X[i])) for i in ben])
    assert mal_p > ben_p
