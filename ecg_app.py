import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

st.title("XWaves - ECG Signal Upload & Analysis")
st.write("Upload an ECG file and get an XWaves-generated report.")

uploaded_file = st.file_uploader("Upload ECG .csv file", type=["csv"])

if uploaded_file is not None:
    try:
        df = pd.read_csv(uploaded_file, header=None)
        st.success("File loaded successfully!")
        st.write("📊 Data Snapshot:")
        st.dataframe(df.head())

        st.write("📈 ECG Signal Plot (Row-wise Mean)")
        ecg_signal = df.mean(axis=1)
        fig, ax = plt.subplots()
        ax.plot(ecg_signal, label='Mean Signal')
        ax.set_xlabel("Time Index")
        ax.set_ylabel("Amplitude")
        ax.legend()
        st.pyplot(fig)

        st.write("🧠 (Optional) Signal Summary Statistics")
        st.write(df.describe())

        # Optional: Add a dummy ML logic
        if ecg_signal.max() > 0.95:
            st.warning("⚠️ High peak detected — further analysis suggested!")
        else:
            st.success("✅ ECG appears within typical range.")

    except Exception as e:
        st.error(f"Error reading file: {e}")
