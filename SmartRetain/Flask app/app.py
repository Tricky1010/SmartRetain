from flask import Flask, request, render_template
import joblib
import numpy as np
import pandas as pd

app = Flask(__name__)
model = joblib.load("model.pkl")

@app.route('/')
def home():
    return render_template("index.html")

@app.route('/predict', methods=["POST"])
def predict():
    tenure = float(request.form['tenure'])
    monthly_charges = float(request.form['monthly_charges'])
    senior_citizen = int(request.form['senior_citizen'])
    contract_one_year = int(request.form['contract_one_year'])
    contract_two_year = int(request.form['contract_two_year'])

    # Exact 30 features as used during training — NO 'contract' column!
    feature_dict = {
        'gender': 0,
        'SeniorCitizen': senior_citizen,
        'Partner': 0,
        'Dependents': 0,
        'tenure': tenure,
        'PhoneService': 0,
        'PaperlessBilling': 0,
        'MonthlyCharges': monthly_charges,
        'TotalCharges': 0,
        'MultipleLines_No phone service': 0,
        'MultipleLines_Yes': 0,
        'InternetService_Fiber optic': 0,
        'InternetService_No': 0,
        'OnlineSecurity_No internet service': 0,
        'OnlineSecurity_Yes': 0,
        'OnlineBackup_No internet service': 0,
        'OnlineBackup_Yes': 0,
        'DeviceProtection_No internet service': 0,
        'DeviceProtection_Yes': 0,
        'TechSupport_No internet service': 0,
        'TechSupport_Yes': 0,
        'StreamingTV_No internet service': 0,
        'StreamingTV_Yes': 0,
        'StreamingMovies_No internet service': 0,
        'StreamingMovies_Yes': 0,
        'Contract_One year': contract_one_year,
        'Contract_Two year': contract_two_year,
        'PaymentMethod_Credit card (automatic)': 0,
        'PaymentMethod_Electronic check': 0,
        'PaymentMethod_Mailed check': 0
    }

    input_df = pd.DataFrame([feature_dict])

    prediction = model.predict(input_df)[0]
    result = "Yes" if prediction == 1 else "No"
    return render_template("index.html", prediction_text=f"Churn Prediction: {result}")

if __name__ == "__main__":
    app.run(debug=True)

