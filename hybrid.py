import pandas as pd
import numpy as np
import joblib
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import IsolationForest
from xgboost import XGBClassifier
from feature_engineering import FeatureEngineer
from sklearn.model_selection import RandomizedSearchCV, cross_val_score
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import roc_auc_score, precision_recall_curve, f1_score, confusion_matrix, classification_report
import matplotlib.pyplot as plt
import shap

# 1. Define features
numerical_features = [
    'account_age_days', 'amount', 'quantity', 'total_value',
    'num_trans_24h', 'num_failed_24h', 'no_of_cards_from_ip',
    'promo_used', 'freq_last_24h', 'amount_last_24h', 'sudden_category_switch',
    'transaction_hour', 'amount_log', 'amount_to_avg', 'new_device_flag', 'hour_sin', 'hour_cos'
]
categorical_features = [
    'payment_method', 'device', 'category'
]

# 2. Preprocessing pipeline
preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numerical_features),
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
    ]
)

# 3. Full pipeline
full_pipeline = Pipeline([
    ('feature_engineering', FeatureEngineer()),
    ('preprocessing', preprocessor)
])

# 4. FraudPredictor class
class FraudPredictor:
    def __init__(self, model_path):
        self.model_data = joblib.load(model_path)
        self.xgb_model = self.model_data['xgb_model']
        self.iso_model = self.model_data['iso_model']
        self.pipeline = self.model_data['pipeline']
        self.iso_thresh = self.model_data['iso_thresh']
        self.feature_names = self.model_data['feature_names']
        # Use the underlying XGBoost model for SHAP
        if hasattr(self.xgb_model, "base_estimator"):
            xgb_for_shap = self.xgb_model.base_estimator
        elif hasattr(self.xgb_model, "estimator"):
            xgb_for_shap = self.xgb_model.estimator
        else:
            xgb_for_shap = self.xgb_model
        self.xgb_explainer = shap.TreeExplainer(xgb_for_shap)
        
    def predict(self, transaction_data):
        processed = self.pipeline.transform(pd.DataFrame([transaction_data]))
        xgb_prob = self.xgb_model.predict_proba(processed)[0][1]
        iso_score = self.iso_model.decision_function(processed)[0]
        # Decision logic
        if xgb_prob > 0.9:
            decision = "FRAUD"
        elif (xgb_prob > 0.6 and iso_score < self.iso_thresh):
            decision = "NEED TO TAKE FEEDBACK"
        else:
            decision = "GENUINE"
        # SHAP explanation
        shap_values = self.xgb_explainer.shap_values(processed)[0]
        indicators = []
        total_impact = sum(np.abs(shap_values))
        for i, val in enumerate(shap_values):
            indicators.append({
                'feature': self.feature_names[i] if i < len(self.feature_names) else f'feature_{i}',
                'value': float(processed[0][i]),
                'impact_percent': round((abs(val)/total_impact)*100, 2) if total_impact else 0.0
            })
        indicators = sorted(indicators, key=lambda x: x['impact_percent'], reverse=True)[:5]
        fraud_pattern = indicators[0]['feature'] if indicators else "unknown"
        return {
            'decision': decision,
            'probability': round(float(xgb_prob), 4),
            'anomaly_score': round(float(iso_score), 4),
            'fraud_indicators': indicators,
            'fraud_pattern': fraud_pattern,
            'thresholds': {
                'xgb_high': 0.9,
                'xgb_feedback': 0.6,
                'iso_threshold': round(float(self.iso_thresh), 4)
            }
        }

# 5. Training and demo code (only runs if this file is executed directly)
if __name__ == "__main__":
    # 5. Load and preprocess data
    def load_and_preprocess_data(file_path):
        df = pd.read_csv(file_path)
        X = df.drop(['fraud_label', 'fraud_pattern', 'Transaction_id', 
                     'billing_address', 'shipping_address', 'ip_address', 'geo_location'], axis=1)
        y = df['fraud_label']
        return X, y, df

    X, y, df = load_and_preprocess_data('fraud.csv')
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, stratify=y, random_state=42
    )
    full_pipeline.fit(X_train)
    X_train_processed = full_pipeline.transform(X_train)
    X_test_processed = full_pipeline.transform(X_test)

    # Hyperparameter tuning for XGBoost
    param_grid = {
        'n_estimators': [200, 300, 500, 700],
        'max_depth': [4, 6, 8, 10],
        'learning_rate': [0.01, 0.05, 0.1],
        'subsample': [0.7, 0.8, 0.9, 1.0],
        'colsample_bytree': [0.7, 0.8, 0.9, 1.0],
        'scale_pos_weight': [30, 40, 45, 50, 60]
    }
    xgb = XGBClassifier(random_state=42, eval_metric='auc')
    search = RandomizedSearchCV(
        xgb, param_grid, n_iter=20, scoring='roc_auc', cv=3, verbose=2, n_jobs=-1
    )
    search.fit(X_train_processed, y_train)
    print("Best params:", search.best_params_)
    xgb_model = search.best_estimator_

    # Cross-validation
    cv_scores = cross_val_score(xgb_model, X_train_processed, y_train, cv=5, scoring='roc_auc')
    print("Cross-validated ROC-AUC:", cv_scores.mean())

    # Feature importance
    cat_feature_names = list(full_pipeline.named_steps['preprocessing'].transformers_[1][1].get_feature_names_out(categorical_features))
    feature_names = numerical_features + cat_feature_names

    importances = xgb_model.feature_importances_
    indices = np.argsort(importances)[::-1]
    print("Feature ranking:")
    for f in range(len(feature_names)):
        print(f"{f + 1}. {feature_names[indices[f]]}: {importances[indices[f]]:.4f}")

    plt.figure(figsize=(10,6))
    plt.title("Feature importances")
    plt.bar(range(len(feature_names)), importances[indices], align="center")
    plt.xticks(range(len(feature_names)), [feature_names[i] for i in indices], rotation=90)
    plt.tight_layout()
    plt.show()

    # Probability calibration
    calibrated_xgb = CalibratedClassifierCV(xgb_model, method='isotonic', cv=3)
    calibrated_xgb.fit(X_train_processed, y_train)
    xgb_model = calibrated_xgb

    # Evaluate with multiple metrics
    y_pred_proba = xgb_model.predict_proba(X_test_processed)[:, 1]
    y_pred = (y_pred_proba > 0.5).astype(int)

    print("ROC-AUC:", roc_auc_score(y_test, y_pred_proba))
    precision, recall, thresholds = precision_recall_curve(y_test, y_pred_proba)
    print("F1-score:", f1_score(y_test, y_pred))
    print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
    print("Classification Report:\n", classification_report(y_test, y_pred))

    iso_model = IsolationForest(
        n_estimators=500,
        contamination=0.03,
        max_samples=256,
        random_state=42
    )
    iso_model.fit(X_train_processed)
    iso_scores = iso_model.decision_function(X_train_processed)
    iso_thresh = np.percentile(iso_scores, 3)

    # Save model as before
    model_package = {
        'xgb_model': xgb_model,
        'iso_model': iso_model,
        'pipeline': full_pipeline,
        'iso_thresh': iso_thresh,
        'feature_names': feature_names
    }
    joblib.dump(model_package, 'hybrid.pkl')

    # Demo prediction
    fraud_predictor = FraudPredictor('hybrid.pkl')
    sample_transaction = {
        'account_age_days': 30,
        'payment_method': 'Credit Card',
        'device': 'Laptop',
        'category': 'Electronics',
        'amount': 35000.0,
        'quantity': 1,
        'total_value': 35000.0,
        'num_trans_24h': 1,
        'num_failed_24h': 0,
        'no_of_cards_from_ip': 1,
        'promo_used': 1,
        'freq_last_24h': 1,
        'amount_last_24h': 35000.0,
        'sudden_category_switch': 0,
        'User_id': 123,
        'TimeStamp': '2024-06-24 14:00:00'
    }
    prediction = fraud_predictor.predict(sample_transaction)
    print(prediction)