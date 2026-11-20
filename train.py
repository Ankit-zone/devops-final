import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


def main():
    df = pd.read_csv("ChurnPrediction.csv")

    # Target & cleaning
    df["Churn"] = df["Churn"].replace({"Yes": 1, "No": 0}).astype(int)
    df["SeniorCitizen"] = df["SeniorCitizen"].replace({1: "Yes", 0: "No"}).astype(str)
    df["TotalCharges"] = df["TotalCharges"].replace(r"\s+", np.nan, regex=True)
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"])

    if "customerID" in df.columns:
        df.drop(["customerID"], axis=1, inplace=True)

    df["TotalCharges"] = df["TotalCharges"].fillna(df["TotalCharges"].median())

    # Binary encoding
    binary_map = {"Yes": 1, "No": 0}
    for col in ["SeniorCitizen", "Partner", "Dependents", "PhoneService"]:
        df[col] = df[col].replace(binary_map).astype(int)

    df["gender"] = df["gender"].replace({"Female": 0, "Male": 1}).astype(int)

    # One-hot encoding
    dummy_cols = [
        "MultipleLines",
        "InternetService",
        "OnlineSecurity",
        "OnlineBackup",
        "DeviceProtection",
        "TechSupport",
        "StreamingTV",
        "StreamingMovies",
        "Contract",
        "PaperlessBilling",
        "PaymentMethod",
    ]
    final_df = pd.get_dummies(columns=dummy_cols, data=df, dtype=int)

    X = final_df.drop(["Churn"], axis=1)
    y = final_df["Churn"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("model", LogisticRegression(max_iter=1000, random_state=42)),
    ])

    pipeline.fit(X_train, y_train)
    preds = pipeline.predict(X_test)

    print(f"Accuracy : {accuracy_score(y_test, preds):.4f}")
    print(f"Precision: {precision_score(y_test, preds):.4f}")
    print(f"Recall   : {recall_score(y_test, preds):.4f}")
    print(f"F1 Score : {f1_score(y_test, preds):.4f}")

    artifact = {
        "model": pipeline,
        "columns": list(X_train.columns),
    }
    joblib.dump(artifact, "churn_pipeline.pkl")
    print("Saved churn_pipeline.pkl successfully.")


if __name__ == "__main__":
    main()