import io
import json
from pathlib import Path

import streamlit as st
from PIL import Image

from model import load_model, colorize_image

st.set_page_config(
    page_title="AI Old Photo Colorizer",
    page_icon="🎨",
    layout="wide"
)

# =====================================================================
# ADD YOUR ACTUAL LINKS HERE (once available), THEN COMMIT TO GITHUB
# =====================================================================
LINKS = {
    "GitHub Repository": "https://github.com/syedahayqagillani/AI-old-photo-colorizer",
    "Dataset (Hugging Face)": "https://huggingface.co/datasets/csr/Image-Colorization",
    "Project Report": "https://docs.google.com/document/d/1qL6AMJHLpE_V71WWi8FYejSNDLNtPoen1VcII-X5WaM/edit?usp=sharingK",
    "Presentation (PPT)": "PASTE_PPT_LINK",
    "YouTube Video": "PASTE_YOUTUBE_LINK",
    "LinkedIn Post": "PASTE_LINKEDIN_LINK",
    "Colab Notebook": "https://colab.research.google.com/drive/1R8RpR85OgF_PqSEZw4ni-evtz94SzskG?usp=sharing",
}


def is_set(url: str) -> bool:
    return url.startswith("http") and "PASTE" not in url and "USERNAME" not in url


@st.cache_resource
def get_model():
    return load_model("weights/colorizer.pth", device="cpu")


# ------------------------------------------------------------------
# Header
# ------------------------------------------------------------------
st.title("🎨 AI Old Photo Colorizer")

st.caption(
    "Deep Learning Semester Project | Convert Black & White Photos into Color Images | "
    "U-Net + ResNet18 Encoder, LAB Color Space"
)

with st.sidebar:
    st.header("🔗 Project Links")

    for name, url in LINKS.items():
        if is_set(url):
            st.markdown(f"- [{name}]({url})")
        else:
            st.markdown(f"- {name}: ⚠️ Link not added yet")


tab_demo, tab_data, tab_model, tab_links = st.tabs(
    ["🎨 Demo", "📦 Dataset & Code", "🧠 Model & Results", "📎 Report / PPT / Video"]
)


# ------------------------------------------------------------------
# Demo
# ------------------------------------------------------------------
with tab_demo:
    st.subheader("Upload a Black & White Image and Bring Its Colors Back")

    up = st.file_uploader(
        "Upload an image (JPG / PNG)",
        type=["jpg", "jpeg", "png"]
    )

    sample_dir = Path("samples/gray")
    samples = sorted(sample_dir.glob("*.jpg")) if sample_dir.exists() else []

    choice = None

    if samples:
        choice = st.selectbox(
            "Or select a sample image",
            ["-"] + [p.name for p in samples]
        )

    strength = st.slider(
        "Color Strength",
        0.5, 2.0, 1.2, 0.1,
        help="A higher value produces more intense colors."
    )

    img = None

    if up is not None:
        img = Image.open(up)

    elif choice and choice != "-":
        img = Image.open(sample_dir / choice)

    if img is None:
        st.info("Please upload an image or select a sample image above.")

    else:
        try:
            model = get_model()

        except Exception as e:
            st.error(
                f"Failed to load model weights. "
                f"Please check weights/colorizer.pth: {e}"
            )
            st.stop()

        with st.spinner("Colorizing your image..."):
            out = colorize_image(model, img, "cpu", strength)

        c1, c2 = st.columns(2)

        c1.image(
            img.convert("L"),
            caption="Input (Black & White)",
            use_container_width=True
        )

        c2.image(
            out,
            caption="AI Colorized Output",
            use_container_width=True
        )

        buf = io.BytesIO()
        Image.fromarray(out).save(buf, format="PNG")

        st.download_button(
            "⬇️ Download Colorized Image",
            buf.getvalue(),
            "colorized.png",
            "image/png"
        )


# ------------------------------------------------------------------
# Dataset & Code
# ------------------------------------------------------------------
with tab_data:
    st.subheader("📦 Dataset")

    st.markdown(f"""
- **Name:** csr/Image-Colorization (Hugging Face)
- **Link:** {LINKS['Dataset (Hugging Face)']}
- **Size:** 6,000 images (~751 MB), License: MIT
- **Format:** Each JPG image contains a grayscale image and its corresponding color image combined side by side.
  We split each image into two halves: **Grayscale Half = Input**, **Color Half = Target**.
- **Preprocessing:** Both images resized to 256x256. The grayscale image's L channel is used as input, while the color image's LAB `a,b` channels are used as targets.
- **Split:** Training 5,400 / Validation 300 / Testing 300
""")

    st.subheader("💻 Code")

    st.markdown(f"GitHub Repository: {LINKS['GitHub Repository']}")
    st.markdown(f"Colab Notebook: {LINKS['Colab Notebook']}")

    if Path("model.py").exists():
        with st.expander("View model.py"):
            st.code(
                Path("model.py").read_text(encoding="utf-8"),
                language="python"
            )


# ------------------------------------------------------------------
# Model & Results
# ------------------------------------------------------------------
with tab_model:
    st.subheader("🧠 Model Architecture")

    st.markdown("""
- **Encoder:** ResNet18 (ImageNet Pretrained)
- **Decoder:** 5 Upsampling Blocks + Skip Connections (U-Net)
- **Input:** L Channel (Brightness) | **Output:** `a,b` Color Channels (tanh)
- **Loss:** Smooth L1 | **Optimizer:** AdamW + OneCycle LR | **Mixed Precision (fp16)**
""")

    st.subheader("📊 Test Results")

    if Path("metrics.json").exists():
        m = json.loads(Path("metrics.json").read_text())
        st.table(m)

    else:
        st.info("Please upload metrics.json to the repository.")

    cols = st.columns(2)

    if Path("loss_curve.png").exists():
        cols[0].image(
            "loss_curve.png",
            caption="Training / Validation Loss",
            use_container_width=True
        )

    if Path("results.png").exists():
        cols[1].image(
            "results.png",
            caption="Input | Prediction | Ground Truth",
            use_container_width=True
        )


# ------------------------------------------------------------------
# Report / PPT / Video
# ------------------------------------------------------------------
with tab_links:
    st.subheader("📎 Project Deliverables")

    for name in [
        "Project Report",
        "Presentation (PPT)",
        "YouTube Video",
        "GitHub Repository",
        "Dataset (Hugging Face)",
        "LinkedIn Post"
    ]:
        url = LINKS[name]

        if is_set(url):
            st.markdown(f"✅ **{name}:** [{url}]({url})")

        else:
            st.markdown(f"⚠️ **{name}:** Link has not been added yet.")

    if is_set(LINKS["YouTube Video"]):
        st.video(LINKS["YouTube Video"])
