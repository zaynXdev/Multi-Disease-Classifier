
import pandas as pd
import numpy as np
import os
from sklearn.preprocessing import MinMaxScaler

DATA_DIR = os.path.join(os.path.dirname(__file__), '..', 'data')
heart_path = os.path.join(DATA_DIR, 'heart.csv')
diabetes_path = os.path.join(DATA_DIR, 'diabetes.csv')


def clean_and_preprocess(df, target_col):
    # Fill missing numeric values with column mean
    for col in df.columns:
        if df[col].dtype in [np.float64, np.int64]:
            if df[col].isnull().sum() > 0:
                df[col].fillna(df[col].mean(), inplace=True)

    # Separate features and target
    X = df.drop(columns=[target_col])
    y = df[target_col]

    # Scale features
    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(X)

    # Return scaled features and target as DataFrame and Series
    X_scaled_df = pd.DataFrame(X_scaled, columns=X.columns)
    y_series = y.reset_index(drop=True)
    return X_scaled_df, y_series


if __name__ == "__main__":
    # Adjust target column names to your datasets
    heart_df = pd.read_csv(heart_path)
    diabetes_df = pd.read_csv(diabetes_path)

    print("Cleaning and preprocessing Heart Disease dataset")
    X_heart, y_heart = clean_and_preprocess(heart_df, target_col='target')  # change target_col if needed
    print(X_heart.head())
    print(y_heart.value_counts())

    print("\nCleaning and preprocessing Diabetes dataset")
    X_diabetes, y_diabetes = clean_and_preprocess(diabetes_df, target_col='Outcome')  # change target_col if needed
    print(X_diabetes.head())
    print(y_diabetes.value_counts())
