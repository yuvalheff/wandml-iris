"""
Iris Species Classification Model Pipeline

Complete pipeline class that combines data processing, feature processing, 
and model prediction for the Iris species classification task.
"""

import pandas as pd
import numpy as np
from typing import Union, Optional

from iris_species_classification.pipeline.data_preprocessing import DataProcessor
from iris_species_classification.pipeline.feature_preprocessing import FeatureProcessor
from iris_species_classification.pipeline.model import ModelWrapper


class ModelPipeline:
    """
    Complete end-to-end pipeline for Iris species classification.
    
    This class combines data preprocessing, feature processing, and model prediction
    into a single interface that can be used for MLflow deployment.
    """
    
    def __init__(self, data_processor: DataProcessor, feature_processor: FeatureProcessor, 
                 model: ModelWrapper):
        """
        Initialize the pipeline components.
        
        Parameters:
        data_processor: Fitted DataProcessor for data preprocessing
        feature_processor: Fitted FeatureProcessor for feature scaling
        model: Trained ModelWrapper for predictions
        """
        self.data_processor = data_processor
        self.feature_processor = feature_processor
        self.model = model
        
        # Validate components are fitted
        if not hasattr(self.data_processor, '_fitted') or not self.data_processor._fitted:
            raise ValueError("DataProcessor must be fitted")
        if not hasattr(self.feature_processor, '_fitted') or not self.feature_processor._fitted:
            raise ValueError("FeatureProcessor must be fitted")
        if not hasattr(self.model, '_fitted') or not self.model._fitted:
            raise ValueError("Model must be fitted")
    
    def predict(self, data: Union[pd.DataFrame, dict, list]) -> np.ndarray:
        """
        Make predictions on input data.
        
        Parameters:
        data: Input data as DataFrame, dict, or list of dicts
        
        Returns:
        np.ndarray: Predicted class labels (encoded)
        """
        # Convert input to DataFrame if needed
        if isinstance(data, dict):
            data = pd.DataFrame([data])
        elif isinstance(data, list) and len(data) > 0 and isinstance(data[0], dict):
            data = pd.DataFrame(data)
        elif not isinstance(data, pd.DataFrame):
            raise ValueError("Input data must be DataFrame, dict, or list of dicts")
        
        # Process through pipeline
        X_processed = self.data_processor.transform_features_only(data)
        X_scaled = self.feature_processor.transform(X_processed)
        predictions = self.model.predict(X_scaled)
        
        return predictions
    
    def predict_proba(self, data: Union[pd.DataFrame, dict, list]) -> np.ndarray:
        """
        Make probability predictions on input data.
        
        Parameters:
        data: Input data as DataFrame, dict, or list of dicts
        
        Returns:
        np.ndarray: Predicted class probabilities
        """
        # Convert input to DataFrame if needed
        if isinstance(data, dict):
            data = pd.DataFrame([data])
        elif isinstance(data, list) and len(data) > 0 and isinstance(data[0], dict):
            data = pd.DataFrame(data)
        elif not isinstance(data, pd.DataFrame):
            raise ValueError("Input data must be DataFrame, dict, or list of dicts")
        
        # Process through pipeline
        X_processed = self.data_processor.transform_features_only(data)
        X_scaled = self.feature_processor.transform(X_processed)
        probabilities = self.model.predict_proba(X_scaled)
        
        return probabilities
    
    def predict_with_labels(self, data: Union[pd.DataFrame, dict, list]) -> dict:
        """
        Make predictions and return results with original class labels.
        
        Parameters:
        data: Input data as DataFrame, dict, or list of dicts
        
        Returns:
        dict: Dictionary containing predictions, probabilities, and class labels
        """
        predictions = self.predict(data)
        probabilities = self.predict_proba(data)
        
        # Convert encoded predictions back to original labels
        original_labels = self.data_processor.inverse_transform_target(
            pd.Series(predictions, name=self.data_processor.target_column)
        )
        
        # Get class labels for probability interpretation
        class_labels = self.data_processor.label_encoder.classes_
        
        return {
            'predictions': predictions.tolist(),
            'predicted_labels': original_labels.tolist(),
            'probabilities': probabilities.tolist(),
            'class_labels': class_labels.tolist()
        }
    
    def get_feature_names(self) -> list:
        """
        Get the feature names used by the pipeline.
        
        Returns:
        list: Feature column names
        """
        return self.data_processor.feature_columns
    
    def get_target_classes(self) -> list:
        """
        Get the target class names.
        
        Returns:
        list: Target class names
        """
        return self.data_processor.label_encoder.classes_.tolist()
