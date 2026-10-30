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

- **Live App:** PASTE_STREAMLIT_LINK
- **Project Report:** PASTE_REPORT_LINK
- **Presentation:** PASTE_PPT_LINK
- **YouTube Video:** PASTE_YOUTUBE_LINK
- **LinkedIn Post:** PASTE_LINKEDIN_LINK
