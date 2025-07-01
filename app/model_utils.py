import pandas as pd
import joblib
from app import db
from app.models import FeedbackCase
from hybrid import full_pipeline, numerical_features, categorical_features
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split, RandomizedSearchCV, cross_val_score
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import IsolationForest
import numpy as np

def retrain_hybrid_model():
    # 1. Load data from FeedbackCase table
    cases = FeedbackCase.query.all()
    if len(cases) < 10:
        print("Not enough data to retrain.")
        return

    df = pd.DataFrame([{
        'account_age_days': c.account_age_days,
        'payment_method': c.payment_method,
        'device': c.device,
        'category': c.category,
        'amount': c.amount,
        'quantity': c.quantity,
        'total_value': c.total_value,
        'num_trans_24h': c.num_trans_24h,
        'num_failed_24h': c.num_failed_24h,
        'no_of_cards_from_ip': c.no_of_cards_from_ip,
        'promo_used': getattr(c, 'promo_used', 0),
        'freq_last_24h': getattr(c, 'freq_last_24h', 0),
        'amount_last_24h': getattr(c, 'amount_last_24h', 0),
        'sudden_category_switch': getattr(c, 'sudden_category_switch', 0),
        'timestamp': c.timestamp,
        'User_id': c.user_id,
        'fraud_label': 1 if c.prediction == "FRAUD" else 0  # Adjust as needed
    } for c in cases])

    # 2. Prepare features and labels
    X = df[numerical_features + categorical_features]
    y = df['fraud_label']

    # 3. Train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, stratify=y, random_state=42
    )

    # 4. Fit pipeline
    full_pipeline.fit(X_train)
    X_train_processed = full_pipeline.transform(X_train)
    X_test_processed = full_pipeline.transform(X_test)

    # 5. Train XGBoost
    param_grid = {
        'n_estimators': [200, 300],
        'max_depth': [4, 6],
        'learning_rate': [0.01, 0.05],
        'subsample': [0.8, 1.0],
        'colsample_bytree': [0.8, 1.0],
        'scale_pos_weight': [30, 50]
    }
    xgb = XGBClassifier(random_state=42, eval_metric='auc')
    search = RandomizedSearchCV(
        xgb, param_grid, n_iter=2, scoring='roc_auc', cv=2, verbose=0, n_jobs=-1
    )
    search.fit(X_train_processed, y_train)
    xgb_model = search.best_estimator_

    # 6. Calibrate
    calibrated_xgb = CalibratedClassifierCV(xgb_model, method='isotonic', cv=2)
    calibrated_xgb.fit(X_train_processed, y_train)
    xgb_model = calibrated_xgb

    # 7. Isolation Forest
    iso_model = IsolationForest(
        n_estimators=100,
        contamination=0.03,
        max_samples=64,
        random_state=42
    )
    iso_model.fit(X_train_processed)
    iso_scores = iso_model.decision_function(X_train_processed)
    iso_thresh = np.percentile(iso_scores, 3)

    # 8. Save model
    model_package = {
        'xgb_model': xgb_model,
        'iso_model': iso_model,
        'pipeline': full_pipeline,
        'iso_thresh': iso_thresh,
        'feature_names': numerical_features + categorical_features
    }
    joblib.dump(model_package, 'models/hybrid.pkl')
    print("Hybrid model retrained and saved.")