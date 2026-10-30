# 🎨 AI Old Photo Colorizer

An AI-powered project that colorizes old black-and-white photographs using deep learning. Developed as a Deep Learning semester project.

## Problem

Predicting the color version of a grayscale image using deep learning techniques (Image Colorization).

## Dataset

- Hugging Face: [Image-Colorization Dataset](https://huggingface.co/datasets/csr/Image-Colorization) (6,000 images, MIT License).
- Each image contains a grayscale image and its corresponding color image combined side by side. We split each image into two parts: the grayscale portion is used as input, while the color portion is used as the target.
- The dataset is not included in this repository. You can access it through the link above.

## Model

- **Architecture:** U-Net with a pretrained ResNet18 encoder.
- **Color Space:** LAB.
- **Input:** L (Lightness) channel.
- **Output:** a and b (Color) channels.
- **Loss Function:** Smooth L1 Loss.

## Repository Structure

```text
app.py                  Streamlit web application
model.py                Model architecture, helper functions, and inference
requirements.txt        Project dependencies
weights/colorizer.pth   Trained model weights
samples/gray/           Sample grayscale images for demonstration
results.png             Model prediction results
loss_curve.png          Training and validation loss curve
metrics.json            Model evaluation metrics
Colorizer_Colab.ipynb   Model training notebook
```

## Make Predictions Using Trained Weights

```python
from PIL import Image
from model import load_model, colorize_image

model = load_model("weights/colorizer.pth")

out = colorize_image(
    model,
    Image.open("samples/gray/sample_00.jpg")
)

# Save the colorized image
Image.fromarray(out).save("out.png")
```

## Run the Application

Install the required dependencies:

```bash
pip install -r requirements.txt
```

Start the Streamlit application:

```bash
streamlit run app.py
```

## Project Links

- **Live App:** https://ai-photo-colorizer.streamlit.app/
- **Project Report:** https://docs.google.com/document/d/1qL6AMJHLpE_V71WWi8FYejSNDLNtPoen1VcII-X5WaM/edit?usp=sharing
- **Presentation:** https://docs.google.com/presentation/d/1mwhqX2P8mfV0t5fqRPVMeeWiRcKlC3h3roIt66kL5dY/edit?usp=sharing
- **YouTube Video:** https://youtu.be/qV6wfc3X9v8
- **LinkedIn Post:** https://lnkd.in/p/dCh3ye9A
- **Colab Notebook:** https://colab.research.google.com/drive/1R8RpR85OgF_PqSEZw4ni-evtz94SzskG?usp=sharing
