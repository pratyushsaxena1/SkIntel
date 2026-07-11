# SkIntel — Skin Lesion Segmentation

A Gradio web app that outlines skin-lesion regions in dermoscopic images using a
U-Net model, served with lightweight ONNX Runtime (no TensorFlow at inference).

## Live app
Deployed on Render's free tier (see below).

## Files
| File | Purpose |
| --- | --- |
| `app.py` | Gradio inference app (loads `skintel_unet.onnx`) |
| `requirements.txt` | Runtime deps: gradio, onnxruntime, opencv, numpy |
| `render.yaml` | Render deployment config (free Python web service) |
| `skintel_unet.onnx` | **Trained model — add this yourself** (see below) |
| `train_and_export.py` | Colab cell: train the U-Net and export `skintel_unet.keras` |
| `convert_to_onnx.py` | Colab cell: convert `skintel_unet.keras` → `skintel_unet.onnx` |
| `main.py` | Original training notebook (Colab export) |

## Getting the model file
The `.onnx` model is produced from the Colab notebook and is **not** checked in
by default:
1. Train the U-Net and export `skintel_unet.keras` (`train_and_export.py`).
2. Convert it to `skintel_unet.onnx` (`convert_to_onnx.py`).
3. Add `skintel_unet.onnx` to the repo root (drag-and-drop in the GitHub UI, or
   `git add skintel_unet.onnx`).

## Deploy on Render (free)
1. Push this repo to GitHub (including `skintel_unet.onnx`).
2. On [render.com](https://render.com): **New → Web Service** → connect this repo.
3. Render reads `render.yaml`: Python runtime, free plan, `pip install -r
   requirements.txt`, `python app.py`.
4. Create the service — it goes live at a public URL.

## Run locally
```bash
pip install -r requirements.txt
python app.py   # opens on http://localhost:7860
```
