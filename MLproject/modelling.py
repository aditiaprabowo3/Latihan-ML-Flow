import mlflow
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import os
import numpy as np
import warnings
import sys

if __name__ == "__main__":
    warnings.filterwarnings("ignore")
    np.random.seed(40)

    # Ambil path file dataset dari argumen atau gunakan default
    file_path = sys.argv[3] if len(sys.argv) > 3 else "MLproject/train_pca.csv"
    data = pd.read_csv(file_path)

    # Pisahkan fitur dan label
    X_train, X_test, y_train, y_test = train_test_split(
        data.drop("Credit_Score", axis=1),
        data["Credit_Score"],
        random_state=42,
        test_size=0.2
    )

    # Ambil parameter dari argumen atau gunakan default
    n_estimators = int(sys.argv[1]) if len(sys.argv) > 1 else 505
    max_depth = int(sys.argv[2]) if len(sys.argv) > 2 else 37

    # Contoh input untuk logging
    input_example = X_train.iloc[:5]

    # Mulai run MLflow
    with mlflow.start_run():
        model = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth)
        model.fit(X_train, y_train)

        # Log model ke MLflow
        mlflow.sklearn.log_model(
            sk_model=model,
            artifact_path="model",
            input_example=input_example
        )
        print("✅ Model berhasil dilog ke MLflow.")

        # Hitung dan log akurasi
        accuracy = model.score(X_test, y_test)
        mlflow.log_metric("accuracy", accuracy)
        print(f"✅ Akurasi model: {accuracy:.4f}")
