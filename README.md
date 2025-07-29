# Multi-Disease Classifier

## Overview

This project implements a **Multi-Disease Classifier** that predicts risks of **Heart Disease** and **Diabetes** using machine learning models (logistic regression). It features:

- Automated relevant feature selection utilities.
- Model training scripts training on top features.
- Prediction modules for real-time inference.
- A Flask web API with a simple interactive frontend for user inputs.
- Instructions for deployment, testing, and troubleshooting.

---

## Project Structure

MultiDiseaseClassifier/
├── data/ # Original datasets (heart.csv, diabetes.csv)
├── models/ # Saved ML model files (*.pkl)
├── src/
│ ├── utils.py # Helper functions for feature selection, data processing
│ ├── train_models.py # Script for training models on selected features
│ ├── predict_models.py # Functions to load models and predict given feature data
│ ├── app.py # Flask API server and frontend app
│ └── templates/
│ └── index.html # HTML form frontend
├── requirements.txt # Python dependencies
└── README.md # This documentation file



## Setup Instructions

### Prerequisites

- Python 3.7 or higher
- pip package installer
- On Windows, Microsoft C++ Build Tools (required for compiling some dependencies like scikit-learn)

### Installation Steps

1. Clone your repository:

git clone https://github.com/yourusername/yourrepo.git
cd yourrepo


2. (Recommended) Create and activate a virtual environment:

- On Windows:

python -m venv venv
venv\Scripts\activate


- On macOS/Linux:

python -m venv venv
source venv/bin/activate


3. Install Python packages:

pip install -r requirements.txt


> **Windows users:** If you encounter errors related to Microsoft Visual C++ Build Tools missing, download and install them here:  
> [https://visualstudio.microsoft.com/visual-cpp-build-tools/](https://visualstudio.microsoft.com/visual-cpp-build-tools/)  
> Select "Desktop development with C++" workload during installation.

---

## Usage

### 1. Train Models

Run the training script that automatically selects the top relevant features and trains logistic regression models:

python src/train_models.py


- Outputs accuracy and classification reports.
- Saves models in `models/` folder (e.g., `heartdisease_logreg.pkl`, `diabetes_logreg.pkl`).

### 2. Run Flask Web API and Frontend

Start the API server:

python src/app.py


- Visit `http://127.0.0.1:5000/` in your browser.
- The page shows a simple form where you can input patient data for both diseases.
- Submit the form to get predicted classifications and probabilities displayed dynamically.

### 3. Use API Programmatically

Send POST requests with JSON payloads to `/predict` endpoint:

- **Endpoint:** `http://127.0.0.1:5000/predict`  
- **Method:** POST  
- **Example JSON Payload:**

{
"heart": {
"age": 63,
"sex": 1,
"cp": 3,
"trestbps": 145,
"chol": 233
},
"diabetes": {
"Pregnancies": 6,
"Glucose": 148,
"BloodPressure": 72,
"SkinThickness": 35,
"Insulin": 0
}
}


- **Response Example:**

{
"heart_disease": {
"prediction": 1,
"probability": [0.2, 0.8]
},
"diabetes": {
"prediction": 0,
"probability": [0.7, 0.3]
}
}



## Code Highlights

- **`utils.py`**: Contains `get_relevant_features()` that selects top N features automatically from datasets using feature importance.
- **`train_models.py`**: Loads dataset, applies feature selection, trains logistic regression models on top features, prints model metrics, saves models.
- **`predict_models.py`**: Loads saved models, validates and prepares input feature dictionaries, predicts classes and probabilities.
- **`app.py`**: Flask app serving both API endpoints (root `/` shows frontend page, `/predict` accepts POST for predictions).
- **`templates/index.html`**: User-friendly HTML form for input, leveraging JavaScript to send data to `/predict` and display results asynchronously.

---

## Troubleshooting & Tips

### TemplateNotFound Error

- Flask looks for `templates/` directory relative to your `app.py` file.
- Ensure your `templates/` folder is inside the same folder as `app.py` (typically `src/templates/index.html`).

### Installing Dependencies on Windows

- Errors mentioning missing Microsoft Visual C++ 14.0 or greater require installing Microsoft C++ Build Tools.
- Download and install from the official page, choosing the C++ development workload.
- Restart your terminal or IDE after installation.
- Upgrade pip tools with:

python -m pip install --upgrade pip setuptools wheel


- For stubborn issues, use precompiled wheels from:  
  [https://www.lfd.uci.edu/~gohlke/pythonlibs/](https://www.lfd.uci.edu/~gohlke/pythonlibs/)

### Git and GitHub

- Use PyCharm Git integration or terminal commands to add, commit, and push your code.
- Typical terminal commands to push your project:

git init # (only once)
git add .
git commit -m "Initial commit with full project"
git remote add origin https://github.com/yourusername/yourrepo.git
git branch -M main
git push -u origin main


---

## Extending with NLP

- NLP code from notebooks can be modularized into a new module (e.g., `nlp_utils.py`).
- Add text-processing endpoints in `app.py` to accept free text inputs.
- Use NLP results as features or separate predictions.
- Update frontend to accept textual symptom descriptions.

---

## Contact & Contribution

Created by Zayn.  
Please report issues, suggest improvements, or open Pull Requests via GitHub.


---

Thank you for using the Multi-Disease Classifier project! Your feedback and contributions are welcome
