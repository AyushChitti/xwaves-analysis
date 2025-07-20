import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

st.set_page_config(page_title="ECG Report Generator", layout="wide")

st.title("🩺 ECG CSV Upload & Report Generator")

# Step 1: Upload CSV
uploaded_file = st.file_uploader("Upload your ECG data (CSV format)", type="csv")

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.success("✅ ECG Data Uploaded Successfully!")

    # Step 2: Preview Data
    st.subheader("📊 Preview of Uploaded ECG Data")
    st.dataframe(df.head())

    # Step 3: ECG Plot
    if 'ecg_signal' in df.columns:
        st.subheader("📈 ECG Signal Visualization")
        plt.figure(figsize=(12, 3))
        plt.plot(df['ecg_signal'][:1000], color='blue', linewidth=1)
        plt.title("ECG Signal (first 1000 samples)")
        plt.xlabel("Time")
        plt.ylabel("Amplitude")
        st.pyplot(plt)

        # Step 4: Basic Heart Rate & Rhythm Analysis (Mock)
        heart_rate = np.random.randint(60, 100)
        rhythm = np.random.choice(["Normal", "Irregular", "Tachycardia", "Bradycardia"])

        st.subheader("📝 Auto-Generated Report")
        st.markdown(f"""
        **Heart Rate**: {heart_rate} bpm  
        **Rhythm Status**: {rhythm}  
        **Report Summary**:  
        The ECG signal appears to have a {rhythm.lower()} rhythm with a heart rate of approximately {heart_rate} bpm.  
        """)
    else:
        st.error("⚠️ 'ecg_signal' column not found in CSV. Please upload a valid ECG dataset with a column named `ecg_signal`.")
else:
    st.info("👈 Upload a CSV file to begin.")
