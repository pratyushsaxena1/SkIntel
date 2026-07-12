---
title: SkIntel
emoji: 🔬
colorFrom: indigo
colorTo: purple
sdk: gradio
sdk_version: 5.9.1
python_version: "3.12"
app_file: app.py
pinned: false
---

# SkIntel: Skin Lesion Segmentation and Malignancy Estimate

Upload a dermoscopic skin-lesion image. A U-Net model outlines the lesion, and a
small CNN estimates whether the lesion may be benign or malignant.

The malignancy output is an educational estimate, not a medical diagnosis.

## Models
- Segmentation: U-Net, served as `skintel_unet.onnx`.
- Classifier: binary CNN (malignant vs benign), served as
  `skintel_classifier.onnx`. Malignant covers akiec, bcc, and mel. Trained on the
  balanced 967-image HAM10000 subset.

Both run under ONNX Runtime, so the app needs no TensorFlow at inference time.

## Files
- `app.py`: Gradio UI.
- `inference.py`: loads both ONNX models and runs segmentation and classification.
- `preprocess.py`: image preprocessing shared by both paths.
- `requirements.txt`: runtime dependencies.
- `skintel_unet.onnx`, `skintel_classifier.onnx`: the trained models.
- `train_classifier.py`, `convert_classifier_to_onnx.py`: Colab scripts to
  retrain and export the classifier.
- `tests/`: preprocessing and inference tests.
