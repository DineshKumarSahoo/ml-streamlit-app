import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.metrics import confusion_matrix, classification_report
from model.train_and_evaluate import train_models

# -------------------------------------------------
# Page Configuration
# -------------------------------------------------
st.set_page_config(
    page_title="Heart Disease ML Classification",
    layout="wide"
)

st.title("❤️ Heart Disease Prediction Dashboard using ML")

st.markdown("""
This project aims to develop a model that predicts whether a patient has heart disease based on various features:
- Uses a default sample dataset (UCI Heart Disease)
- Allows uploading a custom dataset (CSV)
- Automatically trains multiple classification models
- Compares model performance and evaluation metrics
""")

# -------------------------------------------------
# Dataset Handling
# -------------------------------------------------
st.sidebar.header("📂 Dataset Selection")

uploaded_file = st.sidebar.file_uploader(
    "Upload your own CSV file (optional)",
    type=["csv"]
)

# -------------------------------------------------
# Download Sample Dataset
# -------------------------------------------------
st.sidebar.markdown("### 📥 Download Sample Dataset")

with open("heart_disease_uci.csv", "rb") as file:
    st.sidebar.download_button(
        label="Download UCI Heart Disease Dataset",
        data=file,
        file_name="heart_disease_uci.csv",
        mime="text/csv"
    )

# -------------------------
# Default vs Uploaded Dataset
# -------------------------
if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    dataset_name = uploaded_file.name
else:
    df = pd.read_csv("heart_disease_uci.csv")
    dataset_name = "heart_disease_uci.csv (sample dataset)"

st.success(f"📌 Using dataset: **{dataset_name}**")

st.subheader("📄 Dataset Preview")
st.dataframe(df.head(), width="stretch")

# -------------------------------------------------
# Session State Initialization
# -------------------------------------------------
if "trained" not in st.session_state:
    st.session_state.trained = False

if "last_dataset" not in st.session_state:
    st.session_state.last_dataset = None

# -------------------------------------------------
# RESET STATE IF DATASET CHANGES
# -------------------------------------------------
current_dataset_id = dataset_name

if current_dataset_id != st.session_state.last_dataset:
    st.session_state.trained = False
    st.session_state.last_dataset = current_dataset_id

# -------------------------------------------------
# AUTO TRAIN MODELS (ON DATASET LOAD / CHANGE)
# -------------------------------------------------
if not st.session_state.trained:
    with st.spinner("Training models automatically..."):
        results_df, y_test, y_preds = train_models(df)

    st.session_state.results_df = results_df
    st.session_state.y_test = y_test
    st.session_state.y_preds = y_preds
    st.session_state.trained = True

    st.success(f"Models trained successfully using **{dataset_name}**")

# -------------------------------------------------
# SAFETY CHECK (PREVENT UI ERRORS)
# -------------------------------------------------
if not st.session_state.trained:
    st.stop()

# -------------------------------------------------
# FETCH TRAINED RESULTS
# -------------------------------------------------
results_df = st.session_state.results_df
y_test = st.session_state.y_test
y_preds = st.session_state.y_preds

# -------------------------------------------------
# 📊 PERFORMANCE COMPARISON TABLE
# -------------------------------------------------
st.subheader("📊 Model Performance Comparison")

comparison_df = results_df[
    ["Model", "Accuracy", "Precision", "Recall", "F1 Score", "AUC", "MCC"]
].set_index("Model")

st.dataframe(
    comparison_df.round(4),
    width="stretch"
)

st.markdown("---")

# -------------------------------------------------
# ⚙️ MODEL SELECTION
# -------------------------------------------------
st.subheader("⚙️ Model-wise Detailed Evaluation")

model_name = st.selectbox(
    "Select a model to view detailed results",
    comparison_df.index.tolist()
)

selected_metrics = comparison_df.loc[model_name]

# -------------------------------------------------
# 📈 METRICS + 🧮 CONFUSION MATRIX (SIDE BY SIDE)
# -------------------------------------------------
st.subheader("📈 Model Evaluation")

cm = confusion_matrix(
    y_test,
    y_preds[model_name]
)

metrics_col, cm_col = st.columns([2, 1])

# ---------- METRICS ----------
with metrics_col:
    col1, col2, col3 = st.columns(3)
    col1.metric("Accuracy", f"{selected_metrics['Accuracy']:.4f}")
    col2.metric("Precision", f"{selected_metrics['Precision']:.4f}")
    col3.metric("Recall", f"{selected_metrics['Recall']:.4f}")

    col4, col5, col6 = st.columns(3)
    col4.metric("F1 Score", f"{selected_metrics['F1 Score']:.4f}")
    col5.metric("AUC", f"{selected_metrics['AUC']:.4f}")
    col6.metric("MCC", f"{selected_metrics['MCC']:.4f}")

# ---------- CONFUSION MATRIX ----------
with cm_col:
    st.markdown("**Confusion Matrix**")

    fig, ax = plt.subplots(figsize=(3, 3))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        cbar=False,
        square=True,
        ax=ax
    )
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    st.pyplot(fig)
