import pandas as pd
import os
import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    classification_report,
)
from utils import get_relevant_features


DATA_DIR = os.path.join(os.path.dirname(__file__), '..', 'data')
MODELS_DIR = os.path.join(os.path.dirname(__file__), '..', 'models')


heart_data = pd.read_csv(os.path.join(DATA_DIR, 'heart.csv'))
y_heart = heart_data['target']


diabetes_data = pd.read_csv(os.path.join(DATA_DIR, 'diabetes.csv'))
y_diabetes = diabetes_data['Outcome']

def train_and_save(X, y, name):
    print(f"\n--- Training {name} Model ---")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )
    model = LogisticRegression(max_iter=1000)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print(f"Accuracy: {acc:.3f}")
    print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
    print("Classification Report:\n", classification_report(y_test, y_pred))


    os.makedirs(MODELS_DIR, exist_ok=True)
    joblib.dump(model, os.path.join(MODELS_DIR, f"{name.lower()}_logreg.pkl"))
    print(f"{name} model saved!")

if __name__ == "__main__":

    heart_top_features = get_relevant_features('heart', 'target', top_n=5)
    diabetes_top_features = get_relevant_features('diabetes', 'Outcome', top_n=5)


    X_heart = heart_data[heart_top_features]
    X_diabetes = diabetes_data[diabetes_top_features]

    train_and_save(X_heart, y_heart, "HeartDisease")
    train_and_save(X_diabetes, y_diabetes, "Diabetes")
