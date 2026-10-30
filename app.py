import io
import json
from pathlib import Path

import streamlit as st
from PIL import Image

from model import load_model, colorize_image

st.set_page_config(page_title="AI Old Photo Colorizer", page_icon="🎨", layout="wide")

# =====================================================================
# YAHAN APNE ASLI LINKS PASTE KARO (jab ban jayen), phir GitHub pe commit karo
# =====================================================================
LINKS = {
    "GitHub Repository": "https://github.com/USERNAME/ai-old-photo-colorizer",
    "Dataset (Hugging Face)": "https://huggingface.co/datasets/csr/Image-Colorization",
    "Project Report": "PASTE_REPORT_LINK",
    "Presentation (PPT)": "PASTE_PPT_LINK",
    "YouTube Video": "PASTE_YOUTUBE_LINK",
    "LinkedIn Post": "PASTE_LINKEDIN_LINK",
    "Colab Notebook": "PASTE_COLAB_LINK",
}


def is_set(url: str) -> bool:
    return url.startswith("http") and "PASTE" not in url and "USERNAME" not in url


@st.cache_resource
def get_model():
    return load_model("weights/colorizer.pth", device="cpu")


# ------------------------------------------------------------------ header
st.title("🎨 AI Old Photo Colorizer")
st.caption("Deep Learning Semester Project | Black & White photo se colour photo | "
           "U-Net + ResNet18 encoder, LAB colour space")

with st.sidebar:
    st.header("🔗 Project Links")
    for name, url in LINKS.items():
        if is_set(url):
            st.markdown(f"- [{name}]({url})")
        else:
            st.markdown(f"- {name}: ⚠️ link add hona baqi hai")

tab_demo, tab_data, tab_model, tab_links = st.tabs(
    ["🎨 Demo", "📦 Dataset & Code", "🧠 Model & Results", "📎 Report / PPT / Video"]
)

# ------------------------------------------------------------------ demo
with tab_demo:
    st.subheader("Black & White image do, colour wapas lo")
    up = st.file_uploader("Image upload karo (jpg / png)", type=["jpg", "jpeg", "png"])
    sample_dir = Path("samples/gray")
    samples = sorted(sample_dir.glob("*.jpg")) if sample_dir.exists() else []
    choice = None
    if samples:
        choice = st.selectbox("Ya koi sample image select karo", ["-"] + [p.name for p in samples])
    strength = st.slider("Colour strength", 0.5, 2.0, 1.2, 0.1,
                         help="Zyada value = zyada gehra colour")

    img = None
    if up is not None:
        img = Image.open(up)
    elif choice and choice != "-":
        img = Image.open(sample_dir / choice)

    if img is None:
        st.info("Upar se image upload karo ya sample select karo.")
    else:
        try:
            model = get_model()
        except Exception as e:
            st.error(f"Model weights load nahi hue (weights/colorizer.pth check karo): {e}")
            st.stop()
        with st.spinner("Colourize ho rahi hai..."):
            out = colorize_image(model, img, "cpu", strength)
        c1, c2 = st.columns(2)
        c1.image(img.convert("L"), caption="Input (Black & White)", use_container_width=True)
        c2.image(out, caption="AI Colorized Output", use_container_width=True)
        buf = io.BytesIO()
        Image.fromarray(out).save(buf, format="PNG")
        st.download_button("⬇️ Colorized image download karo", buf.getvalue(),
                           "colorized.png", "image/png")

# ------------------------------------------------------------------ dataset & code
with tab_data:
    st.subheader("📦 Dataset")
    st.markdown(f"""
- **Name:** csr/Image-Colorization (Hugging Face)
- **Link:** {LINKS['Dataset (Hugging Face)']}
- **Size:** 6,000 images (~751 MB), license: MIT
- **Format:** har jpg me gray aur colour image ek saath combine hai.
  Humne image ko beech se 2 hisso me tora: **gray half = Input**, **colour half = Target**.
- **Preprocessing:** dono 256x256 resize, gray ka L channel input, colour image ke LAB `a,b` channels target.
- **Split:** Train 5,400 / Validation 300 / Test 300
""")
    st.subheader("💻 Code")
    st.markdown(f"GitHub repo: {LINKS['GitHub Repository']}")
    st.markdown(f"Colab notebook: {LINKS['Colab Notebook']}")
    if Path("model.py").exists():
        with st.expander("model.py dekho"):
            st.code(Path("model.py").read_text(encoding="utf-8"), language="python")

# ------------------------------------------------------------------ model & results
with tab_model:
    st.subheader("🧠 Architecture")
    st.markdown("""
- **Encoder:** ResNet18 (ImageNet pretrained)
- **Decoder:** 5 upsampling blocks + skip connections (U-Net)
- **Input:** L channel (brightness) | **Output:** `a,b` colour channels (tanh)
- **Loss:** Smooth L1 | **Optimizer:** AdamW + OneCycle LR | **Mixed precision (fp16)**
""")
    st.subheader("📊 Test Results")
    if Path("metrics.json").exists():
        m = json.loads(Path("metrics.json").read_text())
        st.table(m)
    else:
        st.info("metrics.json repo me upload karo.")
    cols = st.columns(2)
    if Path("loss_curve.png").exists():
        cols[0].image("loss_curve.png", caption="Training / Validation loss", use_container_width=True)
    if Path("results.png").exists():
        cols[1].image("results.png", caption="Input | Prediction | Ground truth", use_container_width=True)

# ------------------------------------------------------------------ report / ppt / video
with tab_links:
    st.subheader("📎 Project Deliverables")
    for name in ["Project Report", "Presentation (PPT)", "YouTube Video",
                 "GitHub Repository", "Dataset (Hugging Face)", "LinkedIn Post"]:
        url = LINKS[name]
        if is_set(url):
            st.markdown(f"✅ **{name}:** [{url}]({url})")
        else:
            st.markdown(f"⚠️ **{name}:** link abhi add nahi hua")
    if is_set(LINKS["YouTube Video"]):
        st.video(LINKS["YouTube Video"])
