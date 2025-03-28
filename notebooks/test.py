import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix, classification_report
from imblearn.over_sampling import SMOTE
import shap
import warnings
warnings.filterwarnings('ignore')

# Let's assume the dataset is already preprocessed as mentioned
# Load the dataset
def load_data(file_path='telco_customer_churn.csv'):
    """
    Load the preprocessed telco customer churn dataset
    """
    # For demonstration, we'll create a synthetic dataset if file not provided
    try:
        df = pd.read_csv(file_path)
        print(f"Loaded dataset with {df.shape[0]} rows and {df.shape[1]} columns")
    except:
        print("Creating synthetic dataset for demonstration")
        # Create synthetic data matching the description (7043 customers, 20 features)
        np.random.seed(42)
        n_samples = 7043
        
        # Create features similar to telecom churn datasets
        df = pd.DataFrame({
            'gender': np.random.choice(['Male', 'Female'], n_samples),
            'SeniorCitizen': np.random.choice([0, 1], n_samples, p=[0.8, 0.2]),
            'Partner': np.random.choice(['Yes', 'No'], n_samples),
            'Dependents': np.random.choice(['Yes', 'No'], n_samples),
            'tenure': np.random.randint(0, 73, n_samples),
            'PhoneService': np.random.choice(['Yes', 'No'], n_samples, p=[0.9, 0.1]),
            'MultipleLines': np.random.choice(['Yes', 'No', 'No phone service'], n_samples),
            'InternetService': np.random.choice(['DSL', 'Fiber optic', 'No'], n_samples),
            'OnlineSecurity': np.random.choice(['Yes', 'No', 'No internet service'], n_samples),
            'OnlineBackup': np.random.choice(['Yes', 'No', 'No internet service'], n_samples),
            'DeviceProtection': np.random.choice(['Yes', 'No', 'No internet service'], n_samples),
            'TechSupport': np.random.choice(['Yes', 'No', 'No internet service'], n_samples),
            'StreamingTV': np.random.choice(['Yes', 'No', 'No internet service'], n_samples),
            'StreamingMovies': np.random.choice(['Yes', 'No', 'No internet service'], n_samples),
            'Contract': np.random.choice(['Month-to-month', 'One year', 'Two year'], n_samples, p=[0.55, 0.25, 0.2]),
            'PaperlessBilling': np.random.choice(['Yes', 'No'], n_samples),
            'PaymentMethod': np.random.choice(['Electronic check', 'Mailed check', 'Bank transfer', 'Credit card'], n_samples),
            'MonthlyCharges': np.random.uniform(20, 120, n_samples),
            'TotalCharges': np.random.uniform(100, 8000, n_samples),
        })
        
        # Generate target with realistic class imbalance (around 25% churn rate)
        churn_prob = 0.2 + 0.4 * (df['Contract'] == 'Month-to-month') - 0.2 * (df['tenure'] > 40)
        df['Churn'] = np.random.binomial(1, churn_prob, n_samples)
        df['Churn'] = df['Churn'].map({1: 'Yes', 0: 'No'})
        
    # Check for class imbalance
    print(f"Class distribution: {df['Churn'].value_counts(normalize=True).round(3) * 100}%")
    
    return df

def preprocess_data(df):
    """
    Preprocess the data for modeling
    """
    # Make a copy to avoid modifying the original data
    data = df.copy()
    
    # Target variable conversion
    if data['Churn'].dtypes == 'object':
        data['Churn'] = data['Churn'].map({'Yes': 1, 'No': 0})
    
    # Convert categorical variables to dummy variables
    # First, identify categorical columns
    categorical_cols = data.select_dtypes(include=['object']).columns
    categorical_cols = [col for col in categorical_cols if col != 'Churn']
    
    # Create dummy variables
    data = pd.get_dummies(data, columns=categorical_cols, drop_first=True)
    
    # Split features and target
    X = data.drop('Churn', axis=1)
    y = data['Churn']
    
    # Split data into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    # Handle class imbalance using SMOTE (only on training data)
    smote = SMOTE(random_state=42)
    X_train_balanced, y_train_balanced = smote.fit_resample(X_train, y_train)
    
    print(f"Original training set shape: {y_train.value_counts()}")
    print(f"Balanced training set shape: {pd.Series(y_train_balanced).value_counts()}")
    
    # Scale the numerical features
    scaler = StandardScaler()
    numerical_cols = X.select_dtypes(include=['float64', 'int64']).columns
    X_train_balanced[numerical_cols] = scaler.fit_transform(X_train_balanced[numerical_cols])
    X_test[numerical_cols] = scaler.transform(X_test[numerical_cols])
    
    return X_train_balanced, X_test, y_train_balanced, y_test, list(X.columns)

def build_stacked_ensemble():
    """
    Build a stacked ensemble of diverse classifiers
    """
    # Define base classifiers
    base_classifiers = [
        ('rf', RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42)),
        ('gb', GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, random_state=42))
    ]
    
    # Meta-classifier
    meta_clf = LogisticRegression(random_state=42)
    
    return base_classifiers, meta_clf

def train_stacked_ensemble(X_train, y_train, base_classifiers, meta_clf):
    """
    Train the stacked ensemble model
    """
    # Train base classifiers
    base_models = []
    X_meta = np.zeros((X_train.shape[0], len(base_classifiers)))
    
    # Cross-validation for creating meta features
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    
    for i, (name, clf) in enumerate(base_classifiers):
        base_models.append(clf.fit(X_train, y_train))
        
        # Use cross-validation to create meta-features
        cv_preds = np.zeros(X_train.shape[0])
        for train_idx, val_idx in skf.split(X_train, y_train):
            # Train the classifier on the training fold
            clf_cv = clf.__class__(**clf.get_params())
            clf_cv.fit(X_train.iloc[train_idx], y_train.iloc[train_idx])
            
            # Predict on validation fold
            cv_preds[val_idx] = clf_cv.predict_proba(X_train.iloc[val_idx])[:, 1]
            
        X_meta[:, i] = cv_preds
    
    # Train meta-classifier on meta-features
    meta_clf.fit(X_meta, y_train)
    
    return base_models, meta_clf

def predict_with_ensemble(X, base_models, meta_clf, base_classifiers):
    """
    Generate predictions using the stacked ensemble
    """
    # Generate meta-features using base models
    X_meta = np.zeros((X.shape[0], len(base_classifiers)))
    
    for i, (_, _) in enumerate(base_classifiers):
        X_meta[:, i] = base_models[i].predict_proba(X)[:, 1]
    
    # Predict using meta-classifier
    y_pred_proba = meta_clf.predict_proba(X_meta)[:, 1]
    y_pred = meta_clf.predict(X_meta)
    
    return y_pred, y_pred_proba

def evaluate_model(y_true, y_pred, y_pred_proba):
    """
    Evaluate the model performance
    """
    # Calculate performance metrics
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    auc = roc_auc_score(y_true, y_pred_proba)
    
    # Print performance metrics
    print("\nModel Performance:")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1 Score: {f1:.4f}")
    print(f"AUC-ROC: {auc:.4f}")
    
    # Print classification report
    print("\nClassification Report:")
    print(classification_report(y_true, y_pred))
    
    # Plot confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False)
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.tight_layout()
    plt.show()
    
    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'auc': auc
    }

def explain_with_shap(X_train, X_test, base_models):
    """
    Generate SHAP explanations for model predictions
    """
    # We'll use the RandomForest model for SHAP explanations as it's more interpretable
    rf_model = base_models[0]  # Assuming RandomForest is the first model
    
    # Create a SHAP explainer
    explainer = shap.TreeExplainer(rf_model)
    
    # Compute SHAP values for test data (sample for visualization)
    sample_size = min(100, X_test.shape[0])
    X_sample = X_test.sample(sample_size, random_state=42)
    
    # Calculate SHAP values for binary classification
    # For a binary classifier, TreeExplainer sometimes returns a list of arrays instead of a single array
    shap_values = explainer.shap_values(X_sample)
    
    # Check the shape of shap_values to handle different formats
    if isinstance(shap_values, list):
        # For binary classification, use the second element (class 1)
        if len(shap_values) == 2:
            plot_shap_values = shap_values[1]
        else:
            plot_shap_values = shap_values[0]
    else:
        plot_shap_values = shap_values
        
    # Ensure dimensions match before plotting
    if plot_shap_values.shape[1] == X_sample.shape[1]:
        # Summary plot
        plt.figure(figsize=(10, 8))
        shap.summary_plot(plot_shap_values, X_sample, plot_type="bar", show=False)
        plt.title('Feature Importance based on SHAP Values')
        plt.tight_layout()
        plt.show()
        
        # Detailed SHAP visualization for top features
        plt.figure(figsize=(10, 8))
        shap.summary_plot(plot_shap_values, X_sample, show=False)
        plt.title('SHAP Value Distribution by Feature')
        plt.tight_layout()
        plt.show()
    else:
        print(f"Warning: SHAP values shape {plot_shap_values.shape} doesn't match features shape {X_sample.shape}")
        print("Skipping SHAP plots due to dimension mismatch.")
    
    # Return top features based on SHAP values (using feature importances from RF as fallback)
    try:
        if isinstance(shap_values, list) and len(shap_values) == 2:
            feature_importance = np.abs(shap_values[1]).mean(0)
        else:
            feature_importance = np.abs(shap_values).mean(0)
    except:
        print("Using RandomForest feature_importances_ as fallback")
        feature_importance = rf_model.feature_importances_
        
    feature_names = X_test.columns
    importance_df = pd.DataFrame({'Feature': feature_names, 'Importance': feature_importance})
    importance_df = importance_df.sort_values('Importance', ascending=False)
    
    return importance_df, explainer, shap_values

# Alternative approach if the above still fails
def explain_with_shap_alternative(X_train, X_test, base_models):
    """
    Alternative SHAP explanation approach using KernelExplainer
    """
    # We'll use the RandomForest model for SHAP explanations
    rf_model = base_models[0]  # Assuming RandomForest is the first model
    
    # Sample background data for the explainer
    background = shap.sample(X_train, 100)
    
    # Sample test data for explanation
    sample_size = min(50, X_test.shape[0])
    X_sample = X_test.sample(sample_size, random_state=42)
    
    # Define a prediction function that returns probabilities for class 1
    def predict_proba_for_class1(X):
        return rf_model.predict_proba(X)[:, 1]
    
    # Create a KernelExplainer (slower but more robust to different model types)
    explainer = shap.KernelExplainer(predict_proba_for_class1, background)
    
    # Calculate SHAP values (this may take some time)
    shap_values = explainer.shap_values(X_sample)
    
    # Generate plots
    try:
        plt.figure(figsize=(10, 8))
        shap.summary_plot(shap_values, X_sample, plot_type="bar", show=False)
        plt.title('Feature Importance based on SHAP Values')
        plt.tight_layout()
        plt.show()
        
        plt.figure(figsize=(10, 8))
        shap.summary_plot(shap_values, X_sample, show=False)
        plt.title('SHAP Value Distribution by Feature')
        plt.tight_layout()
        plt.show()
    except Exception as e:
        print(f"Error in SHAP plotting: {e}")
    
    # Return top features based on SHAP values
    feature_importance = np.abs(shap_values).mean(0)
    feature_names = X_test.columns
    importance_df = pd.DataFrame({'Feature': feature_names, 'Importance': feature_importance})
    importance_df = importance_df.sort_values('Importance', ascending=False)
    
    return importance_df, explainer, shap_values

def identify_high_risk_customers(X_test, y_pred_proba, threshold=0.7):
    """
    Identify high-risk customers and recommend retention strategies
    """
    # Create a DataFrame with customer features and churn probability
    high_risk_df = X_test.copy()
    high_risk_df['Churn_Probability'] = y_pred_proba
    
    # Identify high-risk customers (high churn probability)
    high_risk_customers = high_risk_df[high_risk_df['Churn_Probability'] > threshold]
    
    print(f"\nIdentified {len(high_risk_customers)} high-risk customers (churn probability > {threshold})")
    
    return high_risk_customers

def develop_retention_strategies(importance_df, high_risk_df, feature_names):
    """
    Develop personalized retention strategies based on SHAP explanations
    """
    # Get top 5 influential features
    top_features = importance_df.head(5)['Feature'].values
    
    # Define strategy templates based on feature categories
    strategies = {
        # Contract-related
        'Contract_Month-to-month': "Offer discounted 1-year or 2-year contracts with incentives",
        'Contract_One year': "Provide early renewal bonuses for upgrading to 2-year contract",
        'Contract_Two year': "Reward loyalty with premium service upgrades",
        
        # Services
        'InternetService_Fiber optic': "Address potential service quality issues, offer network improvements",
        'InternetService_DSL': "Offer upgrade to fiber with first 3 months at DSL price",
        'OnlineSecurity_No': "Provide free trial of online security services",
        'TechSupport_No': "Offer discounted tech support services",
        'OnlineBackup_No': "Provide free cloud backup service for 3 months",
        'DeviceProtection_No': "Bundle device protection with other services at discount",
        
        # Billing
        'PaperlessBilling_Yes': "Offer discount for annual prepayment",
        'PaymentMethod_Electronic check': "Provide discount for switching to automatic credit card payments",
        'MonthlyCharges': "Offer personalized discount based on usage patterns",
        'TotalCharges': "Provide loyalty discount based on tenure and total spend",
        
        # Demographics
        'SeniorCitizen': "Offer senior-specific service bundles with simplified options",
        'tenure': "Recognize customer loyalty with special anniversary offers",
        'Partner_Yes': "Offer family plan upgrades",
        'Dependents_Yes': "Provide family-friendly content packages and parental controls",
    }
    
    # Create general strategies based on top influential features
    print("\nGeneral Retention Strategies Based on Top Influential Features:")
    for feature in top_features:
        base_feature = feature.split('_')[0] if '_' in feature else feature
        if feature in strategies:
            print(f"- For {feature}: {strategies[feature]}")
        elif base_feature in strategies:
            print(f"- For {feature}: {strategies[base_feature]}")
    
    # Sample a few high-risk customers for personalized strategies
    sample_size = min(5, len(high_risk_df))
    if sample_size > 0:
        sampled_customers = high_risk_df.sample(sample_size, random_state=42)
        
        print("\nPersonalized Retention Strategies for Sample High-Risk Customers:")
        for i, (_, customer) in enumerate(sampled_customers.iterrows()):
            print(f"\nCustomer {i+1} (Churn Probability: {customer['Churn_Probability']:.2f}):")
            
            # Identify top 3 features for this customer
            # This is a simplified approach - in a real implementation we would use
            # individual SHAP values for each prediction
            customer_features = []
            for feature in feature_names:
                if feature in customer.index and customer[feature] > 0:
                    if feature in importance_df['Feature'].values:
                        importance = importance_df[importance_df['Feature'] == feature]['Importance'].values[0]
                        customer_features.append((feature, importance, customer[feature]))
            
            # Sort by importance
            customer_features.sort(key=lambda x: x[1], reverse=True)
            
            # Recommend strategies based on top 3 features
            for j, (feature, _, value) in enumerate(customer_features[:3]):
                base_feature = feature.split('_')[0] if '_' in feature else feature
                if feature in strategies:
                    print(f"  {j+1}. {strategies[feature]}")
                elif base_feature in strategies:
                    print(f"  {j+1}. {strategies[base_feature]}")
    
    return top_features

def main():
    # Load and preprocess data
    print("Loading and preprocessing data...")
    df = load_data("G:\LebanseUni\M2\S1\AIDE506-Advanced Machine learining Topics\Project\Data\processed_data.csv")
    X_train, X_test, y_train, y_test, feature_names = preprocess_data(df)
    
    # Build and train stacked ensemble
    print("\nBuilding and training stacked ensemble model...")
    base_classifiers, meta_clf = build_stacked_ensemble()
    base_models, meta_clf = train_stacked_ensemble(X_train, y_train, base_classifiers, meta_clf)
    
    # Evaluate the model
    print("\nEvaluating model performance...")
    y_pred, y_pred_proba = predict_with_ensemble(X_test, base_models, meta_clf, base_classifiers)
    metrics = evaluate_model(y_test, y_pred, y_pred_proba)
    
    # Generate SHAP explanations
    print("\nGenerating SHAP explanations...")
    importance_df, explainer, shap_values = explain_with_shap(X_train, X_test, base_models)
    print("\nTop 10 features by importance:")
    print(importance_df.head(10))
    
    # Identify high-risk customers
    high_risk_customers = identify_high_risk_customers(X_test, y_pred_proba)
    
    # Develop retention strategies
    develop_retention_strategies(importance_df, high_risk_customers, feature_names)
    
    print("\nModel building, evaluation, and actionable insights generation complete.")

if __name__ == "__main__":
    main()