import pandas as pd
import os

DATA_DIR = os.path.join(os.path.dirname(__file__), '..', 'data')

heart_path = os.path.join(DATA_DIR, 'heart.csv')
diabetes_path = os.path.join(DATA_DIR, 'diabetes.csv')

def explore_dataset(path, name):
    print(f"\n--- {name} Dataset Summary ---")
    df = pd.read_csv(path)
    print("\n")
    print("First 10 rows:")
    print(df.head(10))
    print("\n")
    print("\nInfo:")
    print(df.info())

    print("\n")
    print("\nDescribe:")
    print(df.describe())
    print("\n")
    print("\nMissing Values:")
    print(df.isnull().sum())
    print("\n")
    print(f"\nShape: {df.shape}")
    print("\n")
    print(f"Columns: {list(df.columns)}")
    print("\n")

    if 'target' in df.columns:
        print("Unique classes in target:", df['target'].unique())
    elif 'Outcome' in df.columns:
        print("Unique classes in Outcome:", df['Outcome'].unique())
    else:
        print("Check and specify the correct target column name.")
    return df

if __name__ == "__main__":
    heart_df = explore_dataset(heart_path, "Heart Disease")
    diabetes_df = explore_dataset(diabetes_path, "Diabetes")
