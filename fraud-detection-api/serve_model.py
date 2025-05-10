from flask import Flask, request, jsonify
import joblib
import logging
import os

# Initialize the Flask application
app = Flask(__name__)

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Define the base directory (the directory of the current script)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(BASE_DIR, '../models')

# Load models using absolute paths
try:
    models = {
        'xgb_creditcard': joblib.load(os.path.join(MODELS_DIR, 'xgb_creditcard_model.pkl')),
        'xgb_fraud': joblib.load(os.path.join(MODELS_DIR, 'xgb_fraud_model.pkl')),
        'lr_creditcard': joblib.load(os.path.join(MODELS_DIR, 'lr_creditcard_model.pkl')),
        'lr_fraud': joblib.load(os.path.join(MODELS_DIR, 'lr_fraud_model.pkl')),
        'dt_creditcard': joblib.load(os.path.join(MODELS_DIR, 'dt_creditcard_model.pkl')),
        'dt_fraud': joblib.load(os.path.join(MODELS_DIR, 'dt_fraud_model.pkl')),
        'rf_creditcard': joblib.load(os.path.join(MODELS_DIR, 'rf_creditcard_model.pkl')),
        'rf_fraud': joblib.load(os.path.join(MODELS_DIR, 'rf_fraud_model.pkl')),
        'mlp_creditcard': joblib.load(os.path.join(MODELS_DIR, 'mlp_creditcard_model.pkl')),
        'mlp_fraud': joblib.load(os.path.join(MODELS_DIR, 'mlp_fraud_model.pkl'))
    }
except FileNotFoundError as e:
    logger.error(f"Model file not found: {e}")
    raise

# Define a root route to handle GET requests to "/"
@app.route('/', methods=['GET'])
def home():
    return jsonify({'message': 'Welcome to the Fraud Detection API'}), 200

# Define a route to handle favicon requests
@app.route('/favicon.ico')
def favicon():
    return jsonify({'message': 'Favicon not found'}), 404

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json(silent=True)
        if not data:
            return jsonify({'error': 'Invalid or missing JSON payload'}), 400

        model_name = data.get('model_name')
        features = data.get('features')

        # Check if model_name and features are provided
        if not model_name or model_name not in models:
            return jsonify({'error': 'Invalid or missing model_name'}), 400
        if not features:
            return jsonify({'error': 'Features are required'}), 400

        # Select the model and make a prediction
        model = models[model_name]
        prediction = model.predict([features])[0]

        # Log request and prediction
        logger.info(f'Received request with model: {model_name} and features: {features}')
        logger.info(f'Returning prediction: {prediction}')

        return jsonify({'model': model_name, 'prediction': prediction})

    except Exception as e:
        logger.error(f'Error in prediction: {e}')
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)



# from flask import Flask, request, jsonify
# import joblib
# import logging
# import os

# # Initialize the Flask application
# app = Flask(__name__)

# # Set up logging
# logging.basicConfig(level=logging.INFO)
# logger = logging.getLogger(__name__)

# import os
# import joblib

# # Define the base directory (the directory of the current script)
# BASE_DIR = os.path.dirname(os.path.abspath(__file__))
# MODELS_DIR = os.path.join(BASE_DIR, '../models')

# # Load models using absolute paths
# models = {
#     'xgb_creditcard': joblib.load(os.path.join(MODELS_DIR, 'xgb_creditcard_model.pkl')),
#     'xgb_fraud': joblib.load(os.path.join(MODELS_DIR, 'xgb_fraud_model.pkl')),
#     'lr_creditcard': joblib.load(os.path.join(MODELS_DIR, 'lr_creditcard_model.pkl')),
#     'lr_fraud': joblib.load(os.path.join(MODELS_DIR, 'lr_fraud_model.pkl')),
#     'dt_creditcard': joblib.load(os.path.join(MODELS_DIR, 'dt_creditcard_model.pkl')),
#     'dt_fraud': joblib.load(os.path.join(MODELS_DIR, 'dt_fraud_model.pkl')),
#     'rf_creditcard': joblib.load(os.path.join(MODELS_DIR, 'rf_creditcard_model.pkl')),
#     'rf_fraud': joblib.load(os.path.join(MODELS_DIR, 'rf_fraud_model.pkl')),
#     'mlp_creditcard': joblib.load(os.path.join(MODELS_DIR, 'mlp_creditcard_model.pkl')),
#     'mlp_fraud': joblib.load(os.path.join(MODELS_DIR, 'mlp_fraud_model.pkl'))
# }


# @app.route('/predict', methods=['POST'])
# def predict():
#     try:
#         data = request.get_json(force=True)
#         model_name = data.get('model_name')
#         features = data.get('features')

#         # Check if model_name and features are provided
#         if not model_name or model_name not in models:
#             return jsonify({'error': 'Invalid or missing model_name'}), 400
#         if not features:
#             return jsonify({'error': 'Features are required'}), 400

#         # Select the model and make a prediction
#         model = models[model_name]
#         prediction = model.predict([features])[0]

#         # Log request and prediction
#         logger.info(f'Received request with model: {model_name} and features: {features}')
#         logger.info(f'Returning prediction: {prediction}')

#         return jsonify({'model': model_name, 'prediction': prediction})

#     except Exception as e:
#         logger.error(f'Error in prediction: {e}')
#         return jsonify({'error': str(e)}), 500

# if __name__ == '__main__':
#     app.run(host='0.0.0.0', port=5000)
