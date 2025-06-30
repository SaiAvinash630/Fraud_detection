from sklearn.base import BaseEstimator, TransformerMixin
import numpy as np
import pandas as pd

class FeatureEngineer(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        if 'User_id' in X.columns:
            self.user_mean_amount = X.groupby('User_id')['amount'].mean()
        else:
            self.user_mean_amount = None
        return self

    def transform(self, X):
        X = X.copy()
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
        return X