#Evaluate fragrance preduction models

from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, classification_report
import pandas as pd

def evaluate(model, X_test: pd.DataFrame, y_test: pd.Series) -> dict:
    """
    Evaluate model performance.

    Returns:
        dict: metrics
    """
    y_pred = model.predict(X_test)
    metrics = {
        'accuracy': accuracy_score(y_test, y_pred),
        'f1_macro': f1_score(y_test, y_pred, average='macro'),
        'confusion_matrix': confusion_matrix(y_test, y_pred),
        'classification_report': classification_report(y_test, y_pred)
    }
    return metrics