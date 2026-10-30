# 🎨 AI Old Photo Colorizer

Black & White (purani) photos ko deep learning se colour karta hai.
Deep Learning Semester Project.

## Problem
Grayscale image se uska colour version predict karna (image colorization).

## Dataset
- Hugging Face: https://huggingface.co/datasets/csr/Image-Colorization (6,000 images, MIT)
- Har image me gray + colour side-by-side combine hai. Humne usay 2 hisson me tora:
  gray = input, colour = target. (Dataset repo me nahi rakha, link upar hai.)

## Model
U-Net with pretrained ResNet18 encoder, LAB colour space.
Input = L channel, Output = a,b channels. Loss = Smooth L1.

## Repo structure
```
app.py                  Streamlit web app
model.py                model + helper functions + inference
requirements.txt        dependencies
weights/colorizer.pth   trained model weights
samples/gray/           demo images
results.png, loss_curve.png, metrics.json
Colorizer_Colab.ipynb   training notebook
```

## Weights se predict karna
```python
from PIL import Image
from model import load_model, colorize_image

model = load_model("weights/colorizer.pth")
out = colorize_image(model, Image.open("samples/gray/sample_00.jpg"))   # numpy RGB
Image.fromarray(out).save("out.png")
```

## App chalana
```
pip install -r requirements.txt
streamlit run app.py
```

## Links
- Live App: PASTE_STREAMLIT_LINK
- Report: PASTE_REPORT_LINK
- Presentation: PASTE_PPT_LINK
- YouTube: PASTE_YOUTUBE_LINK
- LinkedIn: PASTE_LINKEDIN_LINK
