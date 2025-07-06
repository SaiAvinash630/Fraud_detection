from sklearn.base import BaseEstimator, TransformerMixin
import numpy as np
import pandas as pd
import json

def get_amount(s):
    try:
        products = json.loads(s)
        # Sum all product base_price * quantity for total amount
        return sum(p.get('base_price', 0) * p.get('quantity', 1) for p in products)
    except Exception:
        return 0

def get_quantity(s):
    try:
        products = json.loads(s)
        return sum(p.get('quantity', 1) for p in products)
    except Exception:
        return 0

class FeatureEngineer(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        X = X.copy()
        # Always extract amount and quantity from products
        if 'products' in X.columns:
            X['amount'] = X['products'].apply(get_amount)
            X['quantity'] = X['products'].apply(get_quantity)
        if 'User_id' in X.columns and 'amount' in X.columns:
            self.user_mean_amount = X.groupby('User_id')['amount'].mean()
        else:
            self.user_mean_amount = None
        return self

    def transform(self, X):
        X = X.copy()
        # Always extract amount and quantity from products FIRST
        if 'products' in X.columns:
            X['amount'] = X['products'].apply(get_amount)
            X['quantity'] = X['products'].apply(get_quantity)
        # Now it's safe to use X['amount'] and X['quantity']
        if 'TimeStamp' in X.columns:
            X['transaction_hour'] = pd.to_datetime(X['TimeStamp']).dt.hour
        X['amount_log'] = np.log1p(X['amount'])
        if self.user_mean_amount is not None and 'User_id' in X.columns:
            X['amount_to_avg'] = X.apply(
                lambda row: row['amount'] / self.user_mean_amount.get(row['User_id'], row['amount']),
                axis=1
            )
        else:
            X['amount_to_avg'] = 1.0
        if 'User_id' in X.columns and 'device' in X.columns:
            X['new_device_flag'] = (
                X.groupby('User_id')['device'].apply(lambda s: s != s.shift(1)).astype(int)
            ).fillna(0).values
        else:
            X['new_device_flag'] = 0
        X['hour_sin'] = np.sin(2 * np.pi * X['transaction_hour'] / 24)
        X['hour_cos'] = np.cos(2 * np.pi * X['transaction_hour'] / 24)
        # Ensure new features exist (fill with 0 if missing)
        for col in ['freq_last_24h', 'amount_last_24h', 'sudden_category_switch']:
            if col not in X.columns:
                X[col] = 0
        # Ensure 'category' is extracted from 'products'
        if 'products' in X.columns:
            def get_category(s):
                try:
                    products = json.loads(s)
                    if isinstance(products, list) and len(products) > 0:
                        cats = sorted(set(p['category'] for p in products if 'category' in p))
                        return '|'.join(cats)
                except Exception:
                    pass
                return 'Unknown'
            X['category'] = X['products'].apply(get_category)
            X['multi_category_order'] = X['category'].apply(lambda c: 1 if '|' in c else 0)
        else:
            X['category'] = 'Unknown'
            X['multi_category_order'] = 0
        # Ensure all required features exist
        required_cols = [
            'account_age_days', 'amount', 'quantity', 'total_value',
            'num_trans_24h', 'num_failed_24h', 'no_of_cards_from_ip',
            'promo_used', 'freq_last_24h', 'amount_last_24h', 'sudden_category_switch',
            'transaction_hour', 'amount_log', 'amount_to_avg', 'new_device_flag', 'hour_sin', 'hour_cos',
            'payment_method', 'device', 'category'
        ]
        for col in required_cols:
            if col not in X.columns:
                X[col] = 0 if col in [
                    'account_age_days', 'amount', 'quantity', 'total_value',
                    'num_trans_24h', 'num_failed_24h', 'no_of_cards_from_ip',
                    'promo_used', 'freq_last_24h', 'amount_last_24h', 'sudden_category_switch',
                    'transaction_hour', 'amount_log', 'amount_to_avg', 'new_device_flag', 'hour_sin', 'hour_cos'
                ] else 'Unknown'
        return X