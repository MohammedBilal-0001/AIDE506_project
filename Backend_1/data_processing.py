from flask import Flask, request, jsonify
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
import os
from werkzeug.utils import secure_filename
from flask_cors import CORS

app = Flask(__name__)
CORS(app)
# Configure upload folder
UPLOAD_FOLDER = 'uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Define expected feature names
FEATURE_NAMES = [
    'gender', 'SeniorCitizen', 'Partner', 'Dependents', 'tenure',
    'PhoneService', 'MultipleLines', 'InternetService', 'OnlineSecurity',
    'OnlineBackup', 'DeviceProtection', 'TechSupport', 'StreamingTV',
    'StreamingMovies', 'Contract', 'PaperlessBilling', 'PaymentMethod',
    'MonthlyCharges', 'TotalCharges', 'Churn'
]

# Define categorical mappings
CATEGORICAL_MAPPINGS = {
    'gender': {'Female': 0, 'Male': 1},
    'Partner': {'No': 0, 'Yes': 1},
    'Dependents': {'No': 0, 'Yes': 1},
    'PhoneService': {'No': 0, 'Yes': 1},
    'MultipleLines': {'No': 0, 'No phone service': 1, 'Yes': 2},
    'InternetService': {'DSL': 0, 'Fiber optic': 1, 'No': 2},
    'OnlineSecurity': {'No': 0, 'No internet service': 1, 'Yes': 2},
    'OnlineBackup': {'No': 0, 'No internet service': 1, 'Yes': 2},
    'DeviceProtection': {'No': 0, 'No internet service': 1, 'Yes': 2},
    'TechSupport': {'No': 0, 'No internet service': 1, 'Yes': 2},
    'StreamingTV': {'No': 0, 'No internet service': 1, 'Yes': 2},
    'StreamingMovies': {'No': 0, 'No internet service': 1, 'Yes': 2},
    'Contract': {'Month-to-month': 0, 'One year': 1, 'Two year': 2},
    'PaperlessBilling': {'No': 0, 'Yes': 1},
    'PaymentMethod': {
        'Bank transfer (automatic)': 0,
        'Credit card (automatic)': 1,
        'Electronic check': 2,
        'Mailed check': 3
    }
}
def preprocess_data(df):
    """Perform all preprocessing steps"""
    try:
        # Handle TotalCharges missing values
        if 'TotalCharges' in df.columns:
            df["TotalCharges"] = df["TotalCharges"].replace(" ", pd.NA)
            df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors='coerce')
            df.loc[:, "TotalCharges"] = df["TotalCharges"].fillna(df["TotalCharges"].median())

        # Drop customerID if present
        if 'customerID' in df.columns:
            df.drop(columns=["customerID"], inplace=True)

        # Label encode categorical variables
        label_encoders = {}
        for column in df.columns:
            if df[column].dtype == "object":
                le = LabelEncoder()
                df[column] = le.fit_transform(df[column])
                label_encoders[column] = le

        # Standardize numeric columns
        numeric_cols = ["tenure", "MonthlyCharges", "TotalCharges"]
        numeric_cols = [col for col in numeric_cols if col in df.columns]
        
        if numeric_cols:
            scaler = StandardScaler()
            df[numeric_cols] = scaler.fit_transform(df[numeric_cols])

        return df, label_encoders
     
    except Exception as e:
        print(f"Error in preprocess_data: {e}")
        return None, str(e)
    
def format_data(df):
    """Ensure data matches expected format"""
    try:
        # Apply categorical mappings if not already encoded
        for column, mapping in CATEGORICAL_MAPPINGS.items():
            if column in df.columns and df[column].dtype == 'object':
                df[column] = df[column].map(mapping).fillna(0).astype('int64')

        # Convert numeric columns
        float_cols = ['tenure', 'MonthlyCharges', 'TotalCharges']
        for col in float_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
                df[col] = df[col].fillna(0).astype('float64')

        # Ensure SeniorCitizen is int
        if 'SeniorCitizen' in df.columns:
            df['SeniorCitizen'] = df['SeniorCitizen'].astype('int64')

        # Ensure correct column order (add missing columns with 0)
        for col in FEATURE_NAMES:
            if col not in df.columns:
                df[col] = 0
                
        df = df[FEATURE_NAMES]
        
        return df, None
    except Exception as e:
        return None, str(e)

@app.route('/process_csv', methods=['POST'])
def process_csv():
    try:
        if 'file' not in request.files:
            return jsonify({'error': 'No file uploaded'}), 400
        
        file = request.files['file']
        if file.filename == '':
            return jsonify({'error': 'No selected file'}), 400
        
        if file and file.filename.endswith('.csv'):
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)
            
            # Read and process data
            df = pd.read_csv(filepath)
            processed_df, label_encoders = preprocess_data(df)
            
            if isinstance(label_encoders, str):
                return jsonify({'error': label_encoders}), 400
                
            formatted_df, error = format_data(processed_df)  # Use processed_df instead of df
            if error:
                return jsonify({'error': error}), 400

                # Convert data to native Python types
            formatted_df = formatted_df.astype(object).where(pd.notnull(formatted_df), None)
            # sample_data = formatted_df.head().to_dict(orient='records')
            sample_data = formatted_df.to_dict(orient='records')

            # Convert numpy types in label encoders
            label_mappings = {}
            for col, le in label_encoders.items():
                label_mappings[col] = {
                    'classes': le.classes_.tolist(),
                    'mapping': {k: int(v) for k, v in zip(le.classes_, le.transform(le.classes_))}
                }

            response = {
                'message': 'File processed successfully',
                'row_count': len(formatted_df),
                'columns': formatted_df.columns.tolist(),  # Ensure this is a list
                'sample_data': sample_data,  # Key must match frontend expectation
                'label_mappings': label_mappings
            }
            
            return jsonify(response), 200
        return jsonify({'error': 'Invalid file type'}), 400
        
    except Exception as e:
        app.logger.error(f"Error processing file: {str(e)}")
        return jsonify({'error': f'Processing error: {str(e)}'}), 500
    
if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5001))
    app.run(host='0.0.0.0', port=port, debug=True)
