import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from typing import Dict, Any, List, Optional, Tuple
import os
from pathlib import Path

from sklearn.metrics import (
    roc_auc_score, accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report, roc_curve, auc
)
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.preprocessing import label_binarize

from iris_species_classification.config import ModelEvalConfig


class ModelEvaluator:
    def __init__(self, config: ModelEvalConfig):
        self.config: ModelEvalConfig = config
        self.app_color_palette = [
            'rgba(99, 110, 250, 0.8)',   # Blue
            'rgba(239, 85, 59, 0.8)',    # Red/Orange-Red
            'rgba(0, 204, 150, 0.8)',    # Green
            'rgba(171, 99, 250, 0.8)',   # Purple
            'rgba(255, 161, 90, 0.8)',   # Orange
            'rgba(25, 211, 243, 0.8)',   # Cyan
            'rgba(255, 102, 146, 0.8)',  # Pink
            'rgba(182, 232, 128, 0.8)',  # Light Green
            'rgba(255, 151, 255, 0.8)',  # Magenta
            'rgba(254, 203, 82, 0.8)'    # Yellow
        ]

    def evaluate_model(self, model, X_test: pd.DataFrame, y_test: pd.Series, 
                      class_labels: List[str], plots_dir: str) -> Dict[str, Any]:
        """
        Comprehensive model evaluation with metrics and plots.
        
        Parameters:
        model: Trained model wrapper
        X_test: Test features
        y_test: Test target (encoded)
        class_labels: Original class labels for interpretation
        plots_dir: Directory to save plots
        
        Returns:
        Dict containing all evaluation metrics and results
        """
        # Make predictions
        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test)
        
        # Calculate primary metrics
        macro_auc = roc_auc_score(y_test, y_prob, multi_class='ovr', average='macro')
        accuracy = accuracy_score(y_test, y_pred)
        precision_macro = precision_score(y_test, y_pred, average='macro')
        recall_macro = recall_score(y_test, y_pred, average='macro')
        f1_macro = f1_score(y_test, y_pred, average='macro')
        
        # Per-class metrics
        precision_per_class = precision_score(y_test, y_pred, average=None)
        recall_per_class = recall_score(y_test, y_pred, average=None)
        f1_per_class = f1_score(y_test, y_pred, average=None)
        
        # Confusion matrix
        cm = confusion_matrix(y_test, y_pred)
        
        # Classification report
        clf_report = classification_report(y_test, y_pred, 
                                         target_names=class_labels, 
                                         output_dict=True)
        
        # Create plots
        self._create_confusion_matrix_plot(cm, class_labels, plots_dir)
        self._create_roc_curves_plot(y_test, y_prob, class_labels, plots_dir)
        self._create_probability_distribution_plot(y_prob, y_test, class_labels, plots_dir)
        
        # Feature importance if available
        feature_importance = None
        if hasattr(model, 'get_feature_importance'):
            feature_importance = model.get_feature_importance()
            if feature_importance is not None:
                self._create_feature_importance_plot(feature_importance, 
                                                   list(X_test.columns), plots_dir)
        
        return {
            'primary_metric': self.config.primary_metric,
            'macro_auc': macro_auc,
            'accuracy': accuracy,
            'precision_macro': precision_macro,
            'recall_macro': recall_macro,
            'f1_macro': f1_macro,
            'precision_per_class': precision_per_class.tolist(),
            'recall_per_class': recall_per_class.tolist(),
            'f1_per_class': f1_per_class.tolist(),
            'confusion_matrix': cm.tolist(),
            'classification_report': clf_report,
            'feature_importance': feature_importance.tolist() if feature_importance is not None else None,
            'class_labels': class_labels
        }

    def cross_validate_model(self, model, X_train: pd.DataFrame, y_train: pd.Series) -> Dict[str, Any]:
        """
        Perform cross-validation evaluation.
        
        Parameters:
        model: Model wrapper to evaluate
        X_train: Training features
        y_train: Training target
        
        Returns:
        Dict containing cross-validation results
        """
        cv = StratifiedKFold(n_splits=self.config.cv_folds, 
                           shuffle=True, 
                           random_state=self.config.random_state)
        
        # Cross-validation scores
        scoring_metric = 'roc_auc_ovr_weighted' if self.config.primary_metric == 'macro_auc' else self.config.primary_metric
        cv_scores = cross_val_score(model.model, X_train, y_train, 
                                  cv=cv, scoring=scoring_metric)
        
        return {
            'cv_scores': cv_scores.tolist(),
            'cv_mean': cv_scores.mean(),
            'cv_std': cv_scores.std(),
            'cv_method': self.config.cv_method,
            'n_splits': self.config.cv_folds
        }

    def _create_confusion_matrix_plot(self, cm: np.ndarray, class_labels: List[str], plots_dir: str):
        """Create confusion matrix heatmap plot."""
        fig = go.Figure(data=go.Heatmap(
            z=cm,
            x=class_labels,
            y=class_labels,
            colorscale='Purples',
            showscale=True,
            text=cm,
            texttemplate="%{text}",
            textfont={"size": 12}
        ))
        
        fig.update_layout(
            title="Confusion Matrix",
            xaxis_title="Predicted Label",
            yaxis_title="True Label",
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            font=dict(color='#8B5CF6', size=12),
            title_font=dict(color='#7C3AED', size=16),
            xaxis=dict(
                tickfont=dict(color='#8B5CF6', size=11),
                title_font=dict(color='#7C3AED', size=12)
            ),
            yaxis=dict(
                tickfont=dict(color='#8B5CF6', size=11),
                title_font=dict(color='#7C3AED', size=12)
            )
        )
        
        fig.write_html(f"{plots_dir}/confusion_matrix.html", 
                      include_plotlyjs=True, config={'responsive': True, 'displayModeBar': False})

    def _create_roc_curves_plot(self, y_true: pd.Series, y_prob: np.ndarray, 
                               class_labels: List[str], plots_dir: str):
        """Create ROC curves for each class."""
        # Binarize the output for ROC calculation
        n_classes = len(class_labels)
        y_true_bin = label_binarize(y_true, classes=range(n_classes))
        if n_classes == 2:
            y_true_bin = np.hstack([1 - y_true_bin, y_true_bin])
        
        fig = go.Figure()
        
        # Calculate ROC curve for each class
        for i, label in enumerate(class_labels):
            fpr, tpr, _ = roc_curve(y_true_bin[:, i], y_prob[:, i])
            roc_auc = auc(fpr, tpr)
            
            fig.add_trace(go.Scatter(
                x=fpr, y=tpr,
                mode='lines',
                name=f'{label} (AUC = {roc_auc:.3f})',
                line=dict(color=self.app_color_palette[i % len(self.app_color_palette)])
            ))
        
        # Add diagonal line
        fig.add_trace(go.Scatter(
            x=[0, 1], y=[0, 1],
            mode='lines',
            name='Random Classifier',
            line=dict(dash='dash', color='gray')
        ))
        
        fig.update_layout(
            title="ROC Curves",
            xaxis_title="False Positive Rate",
            yaxis_title="True Positive Rate",
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            font=dict(color='#8B5CF6', size=12),
            title_font=dict(color='#7C3AED', size=16),
            xaxis=dict(
                gridcolor='rgba(139,92,246,0.2)',
                zerolinecolor='rgba(139,92,246,0.3)',
                tickfont=dict(color='#8B5CF6', size=11),
                title_font=dict(color='#7C3AED', size=12)
            ),
            yaxis=dict(
                gridcolor='rgba(139,92,246,0.2)',
                zerolinecolor='rgba(139,92,246,0.3)', 
                tickfont=dict(color='#8B5CF6', size=11),
                title_font=dict(color='#7C3AED', size=12)
            ),
            legend=dict(font=dict(color='#8B5CF6', size=11))
        )
        
        fig.write_html(f"{plots_dir}/roc_curves.html", 
                      include_plotlyjs=True, config={'responsive': True, 'displayModeBar': False})

    def _create_probability_distribution_plot(self, y_prob: np.ndarray, y_true: pd.Series,
                                            class_labels: List[str], plots_dir: str):
        """Create probability distribution plot."""
        fig = make_subplots(rows=1, cols=len(class_labels), 
                           subplot_titles=[f"P({label})" for label in class_labels])
        
        for i, label in enumerate(class_labels):
            probs = y_prob[:, i]
            
            # Create histogram
            fig.add_trace(
                go.Histogram(
                    x=probs,
                    nbinsx=20,
                    name=f'{label}',
                    marker_color=self.app_color_palette[i % len(self.app_color_palette)],
                    opacity=0.7
                ),
                row=1, col=i+1
            )
        
        fig.update_layout(
            title="Predicted Probability Distributions",
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            font=dict(color='#8B5CF6', size=12),
            title_font=dict(color='#7C3AED', size=16),
            showlegend=False
        )
        
        # Update axes
        for i in range(len(class_labels)):
            fig.update_xaxes(
                title_text="Probability", 
                gridcolor='rgba(139,92,246,0.2)',
                tickfont=dict(color='#8B5CF6', size=11),
                title_font=dict(color='#7C3AED', size=12),
                row=1, col=i+1
            )
            fig.update_yaxes(
                title_text="Count" if i == 0 else "",
                gridcolor='rgba(139,92,246,0.2)',
                tickfont=dict(color='#8B5CF6', size=11),
                title_font=dict(color='#7C3AED', size=12),
                row=1, col=i+1
            )
        
        fig.write_html(f"{plots_dir}/probability_distributions.html", 
                      include_plotlyjs=True, config={'responsive': True, 'displayModeBar': False})

    def _create_feature_importance_plot(self, importance: np.ndarray, feature_names: List[str], 
                                       plots_dir: str):
        """Create feature importance bar plot."""
        # Sort features by importance
        indices = np.argsort(importance)[::-1]
        sorted_importance = importance[indices]
        sorted_features = [feature_names[i] for i in indices]
        
        fig = go.Figure(data=[
            go.Bar(
                x=sorted_features,
                y=sorted_importance,
                marker_color=self.app_color_palette[0]
            )
        ])
        
        fig.update_layout(
            title="Feature Importance",
            xaxis_title="Features",
            yaxis_title="Importance",
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            font=dict(color='#8B5CF6', size=12),
            title_font=dict(color='#7C3AED', size=16),
            xaxis=dict(
                gridcolor='rgba(139,92,246,0.2)',
                zerolinecolor='rgba(139,92,246,0.3)',
                tickfont=dict(color='#8B5CF6', size=11),
                title_font=dict(color='#7C3AED', size=12)
            ),
            yaxis=dict(
                gridcolor='rgba(139,92,246,0.2)',
                zerolinecolor='rgba(139,92,246,0.3)', 
                tickfont=dict(color='#8B5CF6', size=11),
                title_font=dict(color='#7C3AED', size=12)
            )
        )
        
        fig.write_html(f"{plots_dir}/feature_importance.html", 
                      include_plotlyjs=True, config={'responsive': True, 'displayModeBar': False})
