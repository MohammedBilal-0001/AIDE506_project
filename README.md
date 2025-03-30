```markdown
# Telco Customer Churn Analysis Project

## Overview
This project focuses on analyzing and predicting customer churn for a telecommunications company using machine learning techniques. By identifying factors that contribute to customer churn, this analysis aims to help the company develop strategies to improve customer retention.

## Dataset
The dataset used in this project is `WA_Fn-UseC_-Telco-Customer-Churn.csv`, which contains information about:
* Customer demographics (gender, senior citizen status, partner status, dependents)
* Service subscriptions (phone service, internet service, security features, streaming services)
* Account information (contract type, billing method, payment method)
* Financial details (monthly charges, total charges)
* Churn status (target variable)

## Data Preprocessing
The following preprocessing steps were performed:
1. **Loading and initial exploration** of the dataset
2. **Handling missing values**:
   * Identified 11 missing values in the TotalCharges column
   * Converted TotalCharges to numeric format
   * Filled missing values with the median
3. **Feature engineering**:
   * Removed non-informative columns (customerID)
   * Applied Label Encoding to categorical variables:
      * Binary features (Yes/No) encoded as 1/0
      * Multi-class features encoded appropriately
      * Detailed mapping documented for reference
4. **Feature scaling**:
   * Applied StandardScaler to numeric features (tenure, MonthlyCharges, TotalCharges)
   * This ensures all features contribute equally to model performance

## Encoding Reference
The categorical variables were encoded as follows:

| Feature          | Encoding                                                               |
|------------------|------------------------------------------------------------------------|
| Gender           | Female: 0, Male: 1                                                     | 
| Partner          | No: 0, Yes: 1                                                          |
| Dependents       | No: 0, Yes: 1                                                          |
| PhoneService     | No: 0, Yes: 1                                                          |
| MultipleLines    | No: 0, No phone service: 1, Yes: 2                                     |
| InternetService  | DSL: 0, Fiber optic: 1, No: 2                                          |
| Online features  | No: 0, No internet service: 1, Yes: 2                                  |
| Contract         | Month-to-month: 0, One year: 1, Two year: 2                            |
| PaperlessBilling | No: 0, Yes: 1                                                          |
| PaymentMethod    | Bank transfer: 0, Credit card: 1, Electronic check: 2, Mailed check: 3 |
| Churn            | No: 0, Yes: 1                                                          |

## Model Development

### Addressing Class Imbalance
To address the class imbalance in the dataset (fewer churned customers than non-churned), we applied SMOTE (Synthetic Minority Over-sampling Technique) to the training data, which creates synthetic examples of the minority class.

### Base Models
Three base models were developed, optimized and evaluated:

1. **Random Forest**
   * Hyperparameter tuning via RandomizedSearchCV
   * Key parameters: n_estimators, max_depth, min_samples_split, min_samples_leaf
   * Results on original data: 81% accuracy
   * Results on SMOTE-balanced data: 75% accuracy, but with improved recall for the positive class (from 51% to 77%)

2. **Logistic Regression**
   * Hyperparameter tuning via GridSearchCV
   * Key parameters: C, penalty, solver
   * Results on original data: 82% accuracy
   * Results on SMOTE-balanced data: 75% accuracy, with improved recall for the positive class (from 58% to 84%)

3. **XGBoost**
   * Hyperparameter tuning via RandomizedSearchCV
   * Key parameters: n_estimators, max_depth, learning_rate, subsample
   * Results on original data: 81% accuracy
   * Results on SMOTE-balanced data: 75% accuracy, with improved recall for the positive class (from 53% to 76%)

### Stacked Model
A stacking ensemble approach was implemented using the three base models:

1. Base models produce probability predictions
2. These probabilities are used as features for a meta-model (XGBoost)
3. The meta-model makes the final prediction

**Results:**
* Accuracy: 76%
* Precision for non-churn (class 0): 93%
* Recall for non-churn (class 0): 74%
* Precision for churn (class 1): 54%
* Recall for churn (class 1): 84%
* F1-score (weighted): 0.78

The stacked model achieves a good balance between detecting churned customers (high recall for class 1) while maintaining reasonable overall accuracy.

## Feature Importance Analysis

SHAP (SHapley Additive exPlanations) analysis was performed to understand which features most influence the churn prediction:

### Key Findings:
1. **Contract type** is the most influential feature
   * Month-to-month contracts are associated with higher churn probability
   * Two-year contracts strongly correlate with customer retention

2. **Monthly charges** show a threshold effect
   * Higher charges increase churn probability
   * There appears to be a threshold where the effect becomes more pronounced

3. **Tenure** is negatively correlated with churn
   * Longer-tenured customers are less likely to churn
   * The effect is nearly linear

4. **Internet service type** influences churn
   * Fiber optic customers show higher churn probability

5. **Online security** is a significant factor
   * Customers without online security are more likely to churn

## Recommended Retention Strategies

Based on the model findings, the following actionable strategies are recommended:

1. **Contract Incentives**
   * Offer special incentives for customers to sign longer-term contracts
   * Example: "Sign a 2-year contract, get 3 months free"

2. **Pricing Strategies**
   * Implement targeted discount tiers for customers identified as high churn risk
   * Maintain monthly charges below the identified threshold for price-sensitive segments

3. **Tenure Rewards**
   * Create a loyalty bonus program that activates at specific tenure milestones
   * Reward long-term customers with special benefits

4. **Service Improvements**
   * Investigate issues with fiber optic service that may be contributing to higher churn
   * Consider bundling online security features at reduced rates to increase adoption

5. **Customer Segmentation**
   * Develop personalized retention strategies for different customer segments based on their predicted churn probability and key influential factors



## Technologies Used
* Python, NumPy, Pandas
* Scikit-learn, XGBoost, SMOTE
* MLflow for experiment tracking
* SHAP for model interpretability
* Matplotlib and Seaborn for visualization

## Conclusion
The stacked ensemble model successfully identifies customers at risk of churning with 84% recall while maintaining good overall performance metrics. The SHAP analysis provides valuable insights into the factors driving customer churn, enabling the development of targeted retention strategies. By implementing the recommended approaches, the telecommunications company can work proactively to improve customer retention and reduce churn.
```
