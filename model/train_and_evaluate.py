import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, roc_auc_score, precision_score,
    recall_score, f1_score, matthews_corrcoef
)
from xgboost import XGBClassifier


def train_models():
    # Load dataset
    df = pd.read_csv("heart_disease_uci.csv")

    # Handle UCI target column
    if "num" in df.columns:
        df["target"] = (df["num"] > 0).astype(int)
        df.drop(columns=["num"], inplace=True)

    # Drop missing values
    df = df.dropna()

    X = df.drop("target", axis=1)
    y = df["target"]

    # Identify column types
    categorical_cols = X.select_dtypes(include=["object", "bool", "category"]).columns
    numeric_cols = X.select_dtypes(include=["int64", "float64"]).columns

    # Preprocessing
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), numeric_cols),
            ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_cols),
        ]
    )

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )

    models = {
        "Logistic Regression": Pipeline([
            ("preprocess", preprocessor),
            ("model", LogisticRegression(max_iter=1000))
        ]),

        "Decision Tree": DecisionTreeClassifier(random_state=42),

        "KNN": Pipeline([
            ("preprocess", preprocessor),
            ("model", KNeighborsClassifier(n_neighbors=5))
        ]),

        "Naive Bayes": Pipeline([
            ("preprocess", preprocessor),
            ("model", GaussianNB())
        ]),

        "Random Forest": RandomForestClassifier(
            n_estimators=200, random_state=42
        ),

        "XGBoost": XGBClassifier(
            eval_metric="logloss",
            random_state=42
        )
    }

    results = []

    for name, model in models.items():
        model.fit(X_train, y_train)

        y_pred = model.predict(X_test)

        # AUC handling
        if hasattr(model, "predict_proba"):
            y_prob = model.predict_proba(X_test)[:, 1]
            auc = roc_auc_score(y_test, y_prob)
        else:
            auc = None

        results.append({
            "Model": name,
            "Accuracy": accuracy_score(y_test, y_pred),
            "AUC": auc,
            "Precision": precision_score(y_test, y_pred),
            "Recall": recall_score(y_test, y_pred),
            "F1 Score": f1_score(y_test, y_pred),
            "MCC": matthews_corrcoef(y_test, y_pred)
        })

    return pd.DataFrame(results)
