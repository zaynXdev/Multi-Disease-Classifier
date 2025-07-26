from flask import Flask, request, jsonify, render_template
from predict_models import predict_heart, predict_diabetes
  # Only needed for external frontends

app = Flask(__name__)



@app.route('/')
def home():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.json
        if not data:
            return jsonify({"error": "No input data provided"}), 400

        # Validate inputs (example: check required fields)
        heart_input = data.get('heart', {})
        diabetes_input = data.get('diabetes', {})

        if not heart_input or not diabetes_input:
            return jsonify({"error": "Missing heart/diabetes inputs"}), 400

        heart_pred, heart_prob = predict_heart(heart_input)
        diabetes_pred, diabetes_prob = predict_diabetes(diabetes_input)

        response = {
            "status": "success",
            "heart_disease": {
                "prediction": int(heart_pred),
                "probability": heart_prob.tolist()
            },
            "diabetes": {
                "prediction": int(diabetes_pred),
                "probability": diabetes_prob.tolist()
            }
        }
        return jsonify(response)

    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    app.run(debug=True)