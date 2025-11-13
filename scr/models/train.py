#Train fragrance family classification models

from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
import pandas as pd
import joblib


def build_pipeline() -> Pipeline:
    """
    Build preprocessing + model pipeline.

    Returns:
        sklearn.pipeline.Pipeline
    """
    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('clf', RandomForestClassifier(n_estimators=100, class_weight='balanced', random_state=42))
    ])
    return pipeline

def train_model(X: pd.DataFrame, y: pd.Series, save_path: str = None) -> Pipeline:
    """
    Train model on features.

    Args:
        X (pd.DataFrame)
        y (pd.Series)
        save_path (str)

    Returns:
        Pipeline
    """
    pipeline = build_pipeline()
    pipeline.fit(X, y)
    if save_path:
        joblib.dump(pipeline, save_path)
    return pipeline