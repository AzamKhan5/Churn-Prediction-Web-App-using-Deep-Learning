from flask import Flask, request, render_template
import numpy as np
import pickle as pk
from tensorflow.keras.models import load_model

app = Flask(__name__)

# Load the correct Keras model
model = load_model('model.h5')  # make sure this file exists
scaler = pk.load(open('scaler.pkl', 'rb'))  # StandardScaler from training

@app.route('/')
def home():
    return render_template('index.html', prediction=None)

@app.route('/predict', methods=['POST'])
def predict():
    # Get form inputs
    CreditScore = int(request.form['CreditScore'])
    Gender = int(request.form['Gender'])
    Age = int(request.form['Age'])
    Tenure = int(request.form['Tenure'])
    Balance = float(request.form['Balance'])
    HasCrCard = int(request.form['HasCrCard'])
    IsActiveMember = int(request.form['IsActiveMember'])
    EstimatedSalary = float(request.form['EstimatedSalary'])
    NumOfProducts = int(request.form['NumOfProducts'])

    # Input feature order must match training
    input_features = [CreditScore, Gender, Age, Tenure, Balance,
                      HasCrCard, IsActiveMember, EstimatedSalary,
                      NumOfProducts]

    # Scale input
    input_scaled = scaler.transform([input_features])

    # Predict (output is probability)
    prediction = model.predict(input_scaled)[0][0]  # float between 0 and 1

    # Optional: print for debugging
    print("Raw prediction probability:", prediction)

    # Apply threshold (default 0.5)
    predicted_class = 1 if prediction >= 0.2 else 0

    return render_template('index.html', prediction=predicted_class)

if __name__ == '__main__':
    app.run(debug=True)
