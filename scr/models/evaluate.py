"""
Model Evaluation for DeepScent
Comprehensive evaluation with metrics and visualizations
"""
import joblib
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, Tuple
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import (
    accuracy_score, 
    f1_score, 
    precision_score,
    recall_score,
    confusion_matrix, 
    classification_report,
    balanced_accuracy_score
)


def evaluate_model(model, X_test: pd.DataFrame, y_test: pd.Series, 
                   save_dir: str = '/files/datascience_project_EG/results/metrics') -> Dict:
    """
    Comprehensive model evaluation
    
    Args:
        model: Trained model
        X_test: Test features
        y_test: Test labels
        save_dir: Directory to save results
        
    Returns:
        Dictionary with all metrics
    """
    print("\n" + "="*70)
    print("MODEL EVALUATION")
    print("="*70)
    
    # Make predictions
    y_pred = model.predict(X_test)
    
    # Calculate metrics
    metrics = {
        'accuracy': accuracy_score(y_test, y_pred),
        'balanced_accuracy': balanced_accuracy_score(y_test, y_pred),
        'f1_macro': f1_score(y_test, y_pred, average='macro'),
        'f1_weighted': f1_score(y_test, y_pred, average='weighted'),
        'precision_macro': precision_score(y_test, y_pred, average='macro', zero_division=0),
        'recall_macro': recall_score(y_test, y_pred, average='macro', zero_division=0),
        'confusion_matrix': confusion_matrix(y_test, y_pred),
        'classification_report': classification_report(y_test, y_pred),
        'classification_report_dict': classification_report(y_test, y_pred, output_dict=True)
    }
    
    # Print results
    print(f"\nOverall Metrics:")
    print(f"  Accuracy:          {metrics['accuracy']:.3f}")
    print(f"  Balanced Accuracy: {metrics['balanced_accuracy']:.3f}")
    print(f"  F1-macro:          {metrics['f1_macro']:.3f}")
    print(f"  F1-weighted:       {metrics['f1_weighted']:.3f}")
    print(f"  Precision-macro:   {metrics['precision_macro']:.3f}")
    print(f"  Recall-macro:      {metrics['recall_macro']:.3f}")
    
    print(f"\nDetailed Classification Report:")
    print(metrics['classification_report'])
    
    # Per-class performance
    print(f"\nPer-Class Performance:")
    report_dict = metrics['classification_report_dict']
    
    # Sort by f1-score
    class_scores = []
    for family, scores in report_dict.items():
        if family not in ['accuracy', 'macro avg', 'weighted avg']:
            if isinstance(scores, dict):
                class_scores.append((
                    family,
                    scores.get('f1-score', 0),
                    scores.get('support', 0)
                ))
    
    class_scores.sort(key=lambda x: x[1], reverse=True)
    
    print(f"\n{'Family':<20} {'F1-Score':>10} {'Support':>10}")
    print("-" * 45)
    for family, f1, support in class_scores[:10]:
        print(f"{family:<20} {f1:>10.3f} {support:>10.0f}")
    
    if len(class_scores) > 10:
        print(f"... and {len(class_scores) - 10} more families")
    
    # Save metrics
    Path(save_dir).mkdir(parents=True, exist_ok=True)
    
    # Save classification report
    report_path = f"{save_dir}/classification_report.txt"
    with open(report_path, 'w') as f:
        f.write("DEEPSCENT MODEL EVALUATION\n")
        f.write("="*70 + "\n\n")
        f.write(f"Accuracy: {metrics['accuracy']:.3f}\n")
        f.write(f"Balanced Accuracy: {metrics['balanced_accuracy']:.3f}\n")
        f.write(f"F1-macro: {metrics['f1_macro']:.3f}\n\n")
        f.write(metrics['classification_report'])
    
    print(f"\n✓ Saved classification report: {report_path}")
    
    return metrics


def plot_confusion_matrix(y_test, y_pred, 
                         save_path: str = '/files/datascience_project_EG/results/figures/confusion_matrix.png',
                         figsize: Tuple[int, int] = (12, 10)):
    """
    Plot and save confusion matrix
    
    Args:
        y_test: True labels
        y_pred: Predicted labels
        save_path: Where to save figure
        figsize: Figure size
    """
    from sklearn.metrics import confusion_matrix
    
    # Get unique classes
    classes = sorted(list(set(y_test) | set(y_pred)))
    
    # Compute confusion matrix
    cm = confusion_matrix(y_test, y_pred, labels=classes)
    
    # Plot
    plt.figure(figsize=figsize)
    
    # Normalize by row (true labels)
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    
    sns.heatmap(cm_normalized, annot=True, fmt='.2f', cmap='Blues',
                xticklabels=classes, yticklabels=classes,
                cbar_kws={'label': 'Proportion'})
    
    plt.title('Confusion Matrix (Normalized by True Label)', fontsize=14, pad=20)
    plt.ylabel('True Label', fontsize=12)
    plt.xlabel('Predicted Label', fontsize=12)
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()
    
    # Save
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"✓ Saved confusion matrix: {save_path}")
    plt.close()


def plot_per_class_metrics(metrics_dict: Dict,
                           save_path: str = '/files/datascience_project_EG/results/figures/per_class_metrics.png'):
    """
    Plot per-class F1, precision, recall
    
    Args:
        metrics_dict: Classification report dictionary
        save_path: Where to save figure
    """
    # Extract per-class metrics
    families = []
    f1_scores = []
    precisions = []
    recalls = []
    supports = []
    
    for family, scores in metrics_dict.items():
        if family not in ['accuracy', 'macro avg', 'weighted avg']:
            if isinstance(scores, dict):
                families.append(family)
                f1_scores.append(scores.get('f1-score', 0))
                precisions.append(scores.get('precision', 0))
                recalls.append(scores.get('recall', 0))
                supports.append(scores.get('support', 0))
    
    # Sort by F1-score
    sorted_indices = np.argsort(f1_scores)[::-1][:15]  # Top 15
    
    families = [families[i] for i in sorted_indices]
    f1_scores = [f1_scores[i] for i in sorted_indices]
    precisions = [precisions[i] for i in sorted_indices]
    recalls = [recalls[i] for i in sorted_indices]
    
    # Plot
    fig, ax = plt.subplots(figsize=(12, 8))
    
    x = np.arange(len(families))
    width = 0.25
    
    ax.bar(x - width, precisions, width, label='Precision', alpha=0.8)
    ax.bar(x, recalls, width, label='Recall', alpha=0.8)
    ax.bar(x + width, f1_scores, width, label='F1-Score', alpha=0.8)
    
    ax.set_xlabel('Fragrance Family', fontsize=12)
    ax.set_ylabel('Score', fontsize=12)
    ax.set_title('Per-Class Performance Metrics (Top 15 Families)', fontsize=14, pad=20)
    ax.set_xticks(x)
    ax.set_xticklabels(families, rotation=45, ha='right')
    ax.legend()
    ax.grid(axis='y', alpha=0.3)
    ax.set_ylim([0, 1.05])
    
    plt.tight_layout()
    
    # Save
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"✓ Saved per-class metrics: {save_path}")
    plt.close()


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Evaluate DeepScent model')
    parser.add_argument('--model', type=str, required=True,
                       help='Path to trained model (.joblib)')
    parser.add_argument('--features', type=str, required=True,
                       help='Path to features CSV')
    parser.add_argument('--output-dir', type=str, default='results',
                       help='Output directory for results')
    
    args = parser.parse_args()
    
    print("="*70)
    print("DEEPSCENT MODEL EVALUATION")
    print("="*70)
    
    # Load model
    print(f"\n📂 Loading model from {args.model}")
    model = joblib.load(args.model)
    
    # Load features
    print(f"📂 Loading features from {args.features}")
    df = pd.read_csv(args.features)
    
    # Prepare data (assuming you have train/test split info)
    # For now, evaluate on all data
    metadata_cols = ['CAS number', 'Principal name', 'Primary descriptor', 
                     'Descriptor 2', 'Descriptor 3', 'fragrance_family', 
                     'all_descriptors', 'SMILES']
    
    feature_cols = [col for col in df.columns if col not in metadata_cols]
    X = df[feature_cols]
    y = df['fragrance_family']
    
    # Evaluate
    metrics = evaluate_model(model, X, y, save_dir=f"{args.output_dir}/metrics")
    
    # Plot confusion matrix
    y_pred = model.predict(X)
    plot_confusion_matrix(y, y_pred, save_path=f"{args.output_dir}/figures/confusion_matrix.png")
    
    # Plot per-class metrics
    plot_per_class_metrics(
        metrics['classification_report_dict'],
        save_path=f"{args.output_dir}/figures/per_class_metrics.png"
    )
    
    print("\n" + "="*70)
    print("EVALUATION COMPLETE!")
    print("="*70)
    print(f"\n📊 Results saved to: {args.output_dir}/")