import pandas as pd
import xgboost as xgb
import os

def load_data(path):
    """Loads the preprocessed data."""
    if not os.path.exists(path):
        raise FileNotFoundError(f"File not found: {path}")
    df = pd.read_csv(path)
    if 'target' in df.columns:
        df = df.rename(columns={'target': 'single_alert_is_regression'})
    return df

def get_xgb_model(scale_pos_weight=1.0):
    """Returns the configured XGBoost model."""
    return xgb.XGBClassifier(
        tree_method='hist',
        device='cuda',
        learning_rate=0.05,
        n_estimators=200,
        max_depth=6,
        subsample=1.0,
        scale_pos_weight=scale_pos_weight,
        random_state=42,
        eval_metric='logloss'
    )
