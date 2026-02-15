# ❤️ Heart Disease Prediction Dashboard using Machine Learning

## 1. Problem Statement

The objective of this project is to build, evaluate, and deploy multiple machine
learning classification models to predict whether a patient has heart disease
based on medical attributes. The application demonstrates an end-to-end ML
workflow including data handling, model training, evaluation, visualization,
and deployment using Streamlit.

---

## 2. Dataset Description

The project uses the **UCI Heart Disease Dataset** as the default sample dataset.
It contains patient medical records with multiple clinical features such as age,
sex, chest pain type, resting blood pressure, cholesterol, and maximum heart rate.
The target variable indicates the presence or absence of heart disease.

- **Number of features:** ≥ 13  
- **Number of instances:** ≥ 900  
- **Type:** Binary classification  

The application also allows users to upload their own CSV dataset. When a new
dataset is uploaded, the models are automatically retrained.

**Sample Dataset:**  
`heart_disease_uci.csv` (provided in the repository and downloadable from the UI)
 Kaggle : https://www.kaggle.com/datasets/redwankarimsony/heart-disease-data

---

## Models Implemented
The following classification models are implemented and trained on the same dataset:

1. Logistic Regression  
2. Decision Tree Classifier  
3. K-Nearest Neighbors (KNN)  
4. Naive Bayes (Gaussian)  
5. Random Forest (Ensemble Model)  
6. XGBoost (Ensemble Model)

---

## 3. Model Performance Comparison

| Model               | Accuracy | AUC    | Precision | Recall | F1     | MCC   |
|---------------------|----------|--------|-----------|--------|--------|-------|
| Logistic Regression | 0.85     | 0.9353 | 0.88      | 0.7857 | 0.8302 | 0.7002|
| Decision Tree       | 0.7333   | 0.7232 | 0.8       | 0.5714 | 0.6667 | 0.4725|
| KNN                 | 0.85     | 0.952  | 0.88      | 0.7857 | 0.8302 | 0.7002|
| Naive Bayes         | 0.65     | 0.9353 | 1         | 0.25   | 0.4    | 0.3885|
| Random Forest       | 0.7833   | 0.9263 | 0.8261    | 0.6786 | 0.7451 | 0.568 |
| XGBoost             | 0.7667   | 0.8984 | 0.85      | 0.6071 | 0.7083 | 0.5433|


## 4. Observations

| Model               | Observation                                                               |
|---------------------|---------------------------------------------------------------------------|
| Logistic Regression | Achieved strong and stable performance with good accuracy, AUC, and MCC, indicating that the medical features are reasonably linearly separable and well-suited for a linear classifier. |
| Decision Tree       | Showed lower accuracy and MCC compared to other models, suggesting overfitting and limited generalization despite being highly interpretable. |
| KNN                 | Performed comparably to Logistic Regression with high AUC, but remains sensitive to feature scaling and local noise in the dataset. |
| Naive Bayes         | Exhibited very high precision but poor recall, indicating strong class bias due to the independence assumption, leading to many missed positive cases. |
| Random Forest       | Demonstrated balanced performance across metrics with improved generalization over a single tree due to ensemble averaging, though not the top performer. |
| XGBoost             | Provided strong non-linear modeling capability with good overall accuracy and MCC, but slightly lower recall compared to the best-performing models. |

## 5. Streamlit Application
    Link : https://ml-app-app-m5yehb9sckpauvdh8nywxx.streamlit.app/

    By default this app displays the information from uploaded sample UCI heart disease data. Please upload heart disease data set to perform model comparison.
