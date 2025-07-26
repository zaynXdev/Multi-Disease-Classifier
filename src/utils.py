

import pandas as pd
import os

DATA_DIR = os.path.join(os.path.dirname(__file__), '..', 'data')


def get_relevant_features(dataset_name, target_col, top_n=5):

    file_path = os.path.join(DATA_DIR, f'{dataset_name}.csv')

    try:
        df = pd.read_csv(file_path)
    except FileNotFoundError:
        print(f"Error: Dataset not found at {file_path}")
        return []

    if target_col not in df.columns:
        print(f"Error: Target column '{target_col}' not found in {dataset_name} dataset.")
        return []


    corr_matrix = df.corr().abs()
    correlations = corr_matrix[target_col].drop(target_col,
                                                errors='ignore')

    top_features = correlations.sort_values(ascending=False).head(top_n)

    return list(top_features.index)


if __name__ == "__main__":

    print("Getting relevant features for Heart Disease...")
    heart_relevant_features = get_relevant_features('heart', 'target', top_n=5)
    print("Heart Disease Top 5 Features:", heart_relevant_features)

    print("\nGetting relevant features for Diabetes...")
    diabetes_relevant_features = get_relevant_features('diabetes', 'Outcome', top_n=5)
    print("Diabetes Top 5 Features:", diabetes_relevant_features)
