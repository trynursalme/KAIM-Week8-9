from flask import Flask, render_template, request
import numpy as np
import joblib

app = Flask(__name__)

# Load your trained model (replace 'model.pkl' with your model file)
model = joblib.load('models/xgb_fraud_model.pkl')

# Load default values for features (you can save these as a numpy array after calculating means)
default_features = np.load('models/feature_defaults.npy')  # An array of 201 default values

@app.route('/')
def home():
    # Only ask for the top 5 most important features (for example)
    important_features_count = 5
    return render_template('index.html', feature_count=important_features_count)

@app.route('/predict', methods=['POST'])
def predict():
    # Get the input from the form (only 5 important features)
    user_features = [float(request.form[f'feature{i}']) for i in range(5)]
    
    # Combine user-provided features with default values
    features = np.array(default_features)  # Start with default values
    features[:5] = user_features  # Replace the first 5 features with user input
    features = features.reshape(1, -1)  # Reshape for the model

    # Make a prediction
    prediction = model.predict(features)

    return render_template('index.html', prediction=prediction[0], feature_count=5)

if __name__ == '__main__':
    app.run(debug=True)
