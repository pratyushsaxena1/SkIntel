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
import gradio as gr

from inference import analyze

DISCLAIMER = (
    "<div style='text-align:center'>Educational estimate only, not a medical "
    "diagnosis. See a dermatologist for any concern.</div>"
)

demo = gr.Interface(
    fn=analyze,
    inputs=gr.Image(type="numpy", label="Skin lesion image"),
    outputs=[
        gr.Image(label="Lesion overlay"),
        gr.Image(label="Segmentation mask"),
        gr.Label(label="Malignancy estimate"),
    ],
    title="SkIntel: Skin Lesion Segmentation",
    description=(
        "<div style='text-align:center'>Upload a dermoscopic image to outline "
        "the lesion region and estimate whether it may be malignant.</div>"
    ),
    article=DISCLAIMER,
    flagging_mode="never",
    css="footer {display: none !important;}",
)

if __name__ == "__main__":
    # Render provides the port via $PORT; default to 7860 for local runs.
    port = int(os.environ.get("PORT", 7860))
    demo.launch(
        server_name="0.0.0.0", server_port=port, ssr_mode=False, show_api=False
    )
