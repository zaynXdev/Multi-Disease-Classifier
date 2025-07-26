

import pandas as pd
import os
import joblib
from utils import get_relevant_features

MODELS_DIR = os.path.join(os.path.dirname(__file__), '..', 'models')


HEART_MODEL_PATH = os.path.join(MODELS_DIR, 'heartdisease_logreg.pkl')
DIABETES_MODEL_PATH = os.path.join(MODELS_DIR, 'diabetes_logreg.pkl')


HEART_TOP_FEATURES = get_relevant_features('heart', 'target', top_n=5)
DIABETES_TOP_FEATURES = get_relevant_features('diabetes', 'Outcome', top_n=5)

def predict_heart(user_input):

    model = joblib.load(HEART_MODEL_PATH)


    input_data = {feat: user_input.get(feat, 0) for feat in HEART_TOP_FEATURES}
    features_df = pd.DataFrame([input_data])
    prediction = model.predict(features_df)
    probability = model.predict_proba(features_df)
    return prediction[0], probability[0]

def predict_diabetes(user_input):

    model = joblib.load(DIABETES_MODEL_PATH)

    input_data = {feat: user_input.get(feat, 0) for feat in DIABETES_TOP_FEATURES}

    features_df = pd.DataFrame([input_data])

    prediction = model.predict(features_df)
    probability = model.predict_proba(features_df)
    return prediction[0], probability[0]

if __name__ == "__main__":

    features_dict_for_heart = {
        HEART_TOP_FEATURES[0]: 0.5,
        HEART_TOP_FEATURES[1]: 1.0,
        HEART_TOP_FEATURES[2]: 0.3,
        HEART_TOP_FEATURES[3]: 0.7,
        HEART_TOP_FEATURES[4]: 0.2,
    }

    features_dict_for_diabetes = {
        DIABETES_TOP_FEATURES[0]: 0.4,
        DIABETES_TOP_FEATURES[1]: 0.8,
        DIABETES_TOP_FEATURES[2]: 0.6,
        DIABETES_TOP_FEATURES[3]: 0.1,
        DIABETES_TOP_FEATURES[4]: 0.9,
    }

    heart_pred, heart_prob = predict_heart(features_dict_for_heart)
    diabetes_pred, diabetes_prob = predict_diabetes(features_dict_for_diabetes)

    print(f"Heart Disease Prediction: {heart_pred}, Probabilities: {heart_prob}")
    print(f"Diabetes Prediction: {diabetes_pred}, Probabilities: {diabetes_prob}")
