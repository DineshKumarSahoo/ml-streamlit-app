import streamlit as st
from model.train_and_evaluate import train_models

st.set_page_config(
    page_title="Heart Disease ML Classification",
    layout="wide"
)

st.title("❤️ Heart Disease Classification – ML Model Deployment")

st.markdown("""
This Streamlit application demonstrates **multiple machine learning classification models**
trained on the **UCI Heart Disease dataset**.
""")

if st.button("🚀 Train Models and Evaluate"):
    with st.spinner("Training models, please wait..."):
        results_df = train_models()

    st.success("Training completed successfully!")

    st.subheader("📊 Model Performance Comparison")
    st.dataframe(results_df.round(4), use_container_width=True)

    st.subheader("🏆 Best Performing Model (Based on F1 Score)")
    best_model = results_df.loc[results_df["F1 Score"].idxmax()]
    st.write(best_model)

st.markdown("""
---
### 🔍 Models Implemented
- Logistic Regression  
- Decision Tree  
- K-Nearest Neighbors  
- Naive Bayes  
- Random Forest (Ensemble)  
- XGBoost (Ensemble)
""")
