import streamlit as st
import pandas as pd
import torch
import torchvision.transforms as transforms
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import os

# Set up Streamlit page
st.set_page_config(page_title="Medical Diagnostics Dashboard", layout="wide")
st.title("🧠 AI Medical Diagnostics Dashboard")

# Sidebar Navigation
app_mode = st.sidebar.radio("Choose Function", ["Dashboard", "X-ray Analyzer", "ECG Analyzer"])

# Image transform
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

# Function to load the model
@st.cache_resource
def load_model():
    try:
        model = torch.load("best_model.pt", map_location=torch.device('cpu'))
        model.eval()
        return model
    except Exception as e:
        st.error(f"Failed to load model: {e}")
        return None

# Dashboard
if app_mode == "Dashboard":
    st.header("📊 Overview")
    st.markdown("""
    Welcome to the **Medical Diagnostics Dashboard**.

    Use the sidebar to:
    - Analyze chest X-ray images for signs of **Pneumonia** using AI.
    - Upload and analyze **ECG signals** (CSV format) for irregular heart patterns.
    """)

# X-ray Analyzer
elif app_mode == "X-ray Analyzer":
    st.header("🩻 Chest X-ray Classification")

    model = load_model()
    if model is not None:
        uploaded_file = st.file_uploader("Upload X-ray Image", type=["jpg", "jpeg", "png"])

        if uploaded_file is not None:
            image = Image.open(uploaded_file).convert("RGB")
            st.image(image, caption="Uploaded X-ray", use_column_width=True)

            input_tensor = transform(image).unsqueeze(0)
            with torch.no_grad():
                output = model(input_tensor)
                prediction = torch.argmax(output, 1).item()
                label = "PNEUMONIA" if prediction == 1 else "NORMAL"
                confidence = torch.nn.functional.softmax(output, dim=1)[0][prediction].item()

            st.success(f"Prediction: **{label}** (Confidence: {confidence:.2f})")

# ECG Analyzer
elif app_mode == "ECG Analyzer":
    st.header("❤️ ECG CSV Analysis")
    uploaded_csv = st.file_uploader("Upload ECG CSV", type="csv")

    if uploaded_csv is not None:
        try:
            df = pd.read_csv(uploaded_csv)
            # Auto-detect column names or assign them manually if needed
            if df.shape[1] == 2:
                df.columns = ["Time", "ECG"]
            else:
                st.error("CSV must contain two columns: Time and ECG signal.")
                st.stop()

            # Show line chart
            st.line_chart(df.set_index("Time"))

            # Basic analysis logic
            std_dev = df["ECG"].std()
            mean_val = df["ECG"].mean()

            if std_dev > 0.1:
                st.error("⚠️ Abnormal ECG pattern detected! Possible Arrhythmia.")
            else:
                st.success("✅ ECG signal appears normal.")

            # Report Section
            st.markdown("#### 📄 ECG Report")
            st.json({
                "Mean Voltage": f"{mean_val:.3f}",
                "Standard Deviation": f"{std_dev:.3f}",
                "Analysis Result": "Abnormal" if std_dev > 0.1 else "Normal"
            })

        except Exception as e:
            st.error(f"Error reading file: {e}")
