# AI-Based Cancer Type Detection using Deep Learning

## Project Description

An AI-based deep learning system for classifying cancer types from histopathology images using a Convolutional Neural Network (CNN). The model is integrated with a Streamlit web application for real-time image prediction and visualization.

## Technologies Used

- Python
- TensorFlow
- Keras
- CNN
- Streamlit
- NumPy
- Matplotlib

## Features

- Upload histopathology images for prediction
- Real-time cancer type classification
- Displays prediction confidence
- Visualizes class probabilities
- Interactive Streamlit web interface

## Cancer Types Detected

The model classifies images into five categories:

1. Colon Adenocarcinoma
2. Colon Benign Tissue
3. Lung Adenocarcinoma
4. Lung Benign Tissue
5. Lung Squamous Cell Carcinoma

## Model

The project uses a trained CNN model for image classification.

The input image is resized to 224 × 224 pixels and normalized before prediction.

## Project Files

- `main.py` – Streamlit application for cancer type prediction
- `trainer.ipynb` – Model training and evaluation notebook
- `confusion_matrix.png` – Confusion matrix showing model classification results
- `README.md` – Project documentation

## How to Run

### 1. Install required libraries

```bash
pip install streamlit tensorflow numpy matplotlib pillow
