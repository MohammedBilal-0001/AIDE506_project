import os
from flask import Flask, request, jsonify
from flask_cors import CORS
import pandas as pd
import joblib
import numpy as np
import shap
import mlflow.pyfunc
import dagshub


dagshub.init(repo_owner='MohammedBilal-0001',repo_name='AIDE506_project',mlflow=True)
# Initialize Flask app
app = Flask(__name__)
CORS(app)

# Load the model
def load_model():
    #model_path = "file:///E:/LU/9_M2_First%20Semester/AIDE505_Machine%20Learning%20Data%20Science%20for%20Production/project_505/notebooks/mlruns/225667749794240171/c3e1c0676f924898bdead3b9e94eba04/artifacts/model"
    xgb_meta_model_url= "runs:/c8ea6a9809514a1bb19f9af3452dec60/model"
    try:
        #model = mlflow.pyfunc.load_model(model_path)
        xgb_meta_model= mlflow.sklearn.load_model(xgb_meta_model_url)
        print("Model loaded successfully")
        return xgb_meta_model
    except Exception as e:
        print(f"Error loading model: {e}")
        return None

model = load_model()

# Define expected feature names
FEATURE_NAMES = [
    'gender', 'SeniorCitizen', 'Partner', 'Dependents', 
    'tenure', 'PhoneService', 'MultipleLines', 'InternetService',
    'OnlineSecurity', 'OnlineBackup', 'DeviceProtection', 
    'TechSupport', 'StreamingTV', 'StreamingMovies',
    'Contract', 'PaperlessBilling', 'PaymentMethod',
    'MonthlyCharges', 'TotalCharges'
]

@app.route('/api/predict', methods=['POST'])
def predict():
    if model is None:
        return jsonify({'error': 'Model not loaded'}), 500
    
    try:
        data = request.json
        
        # Validate input format
        if not isinstance(data, dict):
            return jsonify({'error': 'Input must be a JSON object'}), 400
            
        # Convert to DataFrame and validate features
        try:
            df = pd.DataFrame([data])
            missing_features = set(FEATURE_NAMES) - set(df.columns)
            if missing_features:
                return jsonify({'error': f'Missing features: {missing_features}'}), 400
            df = df[FEATURE_NAMES]
        except Exception as e:
            return jsonify({'error': f'Data validation failed: {str(e)}'}), 400

        
        raw_prediction = model.predict(df)
        prediction_proba = np.array([[1 - raw_prediction[0], raw_prediction[0]]])  

        # Format response
        return jsonify({
            'prediction': "Yes" if prediction_proba[0][1] > 0.5 else "No",
            'churn_probability': float(prediction_proba[0][1]),
            'non_churn_probability': float(prediction_proba[0][0])
        })
    
    except Exception as e:
        return jsonify({'error': f'Prediction error: {str(e)}'}), 500

@app.route('/api/health', methods=['GET'])
def health_check():
    return jsonify({
        'status': 'healthy' if model is not None else 'unhealthy',
        'model_loaded': model is not None
    }), 200 if model is not None else 500

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5002))
    app.run(host='0.0.0.0', port=port, debug=True)