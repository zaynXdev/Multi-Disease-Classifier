
import pandas as pd
import numpy as np
import os
from sklearn.preprocessing import MinMaxScaler

DATA_DIR = os.path.join(os.path.dirname(__file__), '..', 'data')
heart_path = os.path.join(DATA_DIR, 'heart.csv')
diabetes_path = os.path.join(DATA_DIR, 'diabetes.csv')


def clean_and_preprocess(df, target_col):

    for col in df.columns:
        if df[col].dtype in [np.float64, np.int64]:
            if df[col].isnull().sum() > 0:
                df[col].fillna(df[col].mean(), inplace=True)


    X = df.drop(columns=[target_col])
    y = df[target_col]


    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(X)


    X_scaled_df = pd.DataFrame(X_scaled, columns=X.columns)
    y_series = y.reset_index(drop=True)
    return X_scaled_df, y_series


if __name__ == "__main__":

    heart_df = pd.read_csv(heart_path)
    diabetes_df = pd.read_csv(diabetes_path)

    print("Cleaning and preprocessing Heart Disease dataset")
    X_heart, y_heart = clean_and_preprocess(heart_df, target_col='target')
    print(X_heart.head())
    print(y_heart.value_counts())

    print("\nCleaning and preprocessing Diabetes dataset")
    X_diabetes, y_diabetes = clean_and_preprocess(diabetes_df, target_col='Outcome')
    print(X_diabetes.head())
    print(y_diabetes.value_counts())
