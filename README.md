# SkIntel

Skin lesion segmentation web app. Upload a dermoscopic image and a U-Net model
outlines the lesion region. Built with Gradio and ONNX Runtime.

## Files

- `app.py`: Gradio app that loads `skintel_unet.onnx`
- `requirements.txt`: dependencies
- `render.yaml`: Render deploy config (free tier)
- `skintel_unet.onnx`: trained model
- `train_and_export.py`: Colab script to train and export the model
- `convert_to_onnx.py`: Colab script to convert the Keras model to ONNX
- `main.py`: original training notebook

## Deploy on Render

1. Push this repo to GitHub.
2. On render.com, create a new Web Service from this repo.
3. Build command: `pip install -r requirements.txt`
4. Start command: `python app.py`
5. Pick the Free instance type and create the service.

## Run locally

```bash
pip install -r requirements.txt
python app.py
```
