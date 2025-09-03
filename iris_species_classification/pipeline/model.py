import pandas as pd
import numpy as np
import pickle
from typing import Optional

from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.base import BaseEstimator, ClassifierMixin

from iris_species_classification.config import ModelConfig


class ModelWrapper(BaseEstimator, ClassifierMixin):
    def __init__(self, config: ModelConfig):
        self.config: ModelConfig = config
        self.model = None
        self._fitted = False
        self._init_model()

    def _init_model(self):
        """Initialize the model based on config."""
        model_type_lower = self.config.model_type.lower()
        if model_type_lower == 'randomforest' or model_type_lower == 'randomforestclassifier':
            self.model = RandomForestClassifier(**self.config.model_params)
        elif model_type_lower == 'extratrees' or model_type_lower == 'extratreesclassifier':
            self.model = ExtraTreesClassifier(**self.config.model_params)
        else:
            raise ValueError(f"Unsupported model type: {self.config.model_type}")

    def fit(self, X: pd.DataFrame, y: pd.Series):
        """
        Fit the classifier to the training data.

        Parameters:
        X: Training features.
        y: Target labels.

        Returns:
        self: Fitted classifier.
        """
        if self.model is None:
            raise ValueError("Model not initialized")
        
        # Fit the model
        self.model.fit(X, y)
        self._fitted = True
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """
        Predict class labels for the input features.

        Parameters:
        X: Input features to predict.

        Returns:
        np.ndarray: Predicted class labels.
        """
        if not self._fitted:
            raise ValueError("Model must be fitted before prediction")
        
        return self.model.predict(X)

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """
        Predict class probabilities for the input features.

        Parameters:
        X: Input features to predict probabilities.

        Returns:
        np.ndarray: Predicted class probabilities.
        """
        if not self._fitted:
            raise ValueError("Model must be fitted before prediction")
        
        return self.model.predict_proba(X)

    def get_feature_importance(self) -> Optional[np.ndarray]:
        """
        Get feature importance from the trained model.

        Returns:
        np.ndarray: Feature importance scores if available.
        """
        if not self._fitted:
            raise ValueError("Model must be fitted before accessing feature importance")
        
        if hasattr(self.model, 'feature_importances_'):
            return self.model.feature_importances_
        else:
            return None

    def save(self, path: str) -> None:
        """
        Save the model wrapper as an artifact.

        Parameters:
        path (str): The file path to save the model.
        """
        with open(path, 'wb') as f:
            pickle.dump(self, f)

    def load(self, path: str) -> 'ModelWrapper':
        """
        Load the model wrapper from a saved artifact.

        Parameters:
        path (str): The file path to load the model from.

        Returns:
        ModelWrapper: The loaded model wrapper.
        """
        with open(path, 'rb') as f:
            return pickle.load(f)