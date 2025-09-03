from typing import Optional
import pandas as pd
import pickle

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import StandardScaler

from iris_species_classification.config import FeaturesConfig


class FeatureProcessor(BaseEstimator, TransformerMixin):
    def __init__(self, config: FeaturesConfig):
        self.config: FeaturesConfig = config
        self.scaler = StandardScaler()
        self._fitted = False

    def _create_ratio_features(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Create ratio features based on configuration.
        
        Parameters:
        X (pd.DataFrame): Input features
        
        Returns:
        pd.DataFrame: Features with ratio features added
        """
        X_with_ratios = X.copy()
        
        if self.config.create_ratio_features and self.config.ratio_features:
            for ratio_config in self.config.ratio_features:
                if ratio_config.numerator in X.columns and ratio_config.denominator in X.columns:
                    # Create ratio feature, handle division by zero
                    denominator = X[ratio_config.denominator]
                    ratio_values = X[ratio_config.numerator] / denominator.replace(0, 1e-8)  # Avoid division by zero
                    X_with_ratios[ratio_config.name] = ratio_values
                else:
                    raise ValueError(f"Required columns {ratio_config.numerator} or {ratio_config.denominator} not found for ratio feature {ratio_config.name}")
        
        return X_with_ratios

    def fit(self, X: pd.DataFrame, y: Optional[pd.Series] = None) -> 'FeatureProcessor':
        """
        Fit the feature processor to the data.

        Parameters:
        X (pd.DataFrame): The input features.
        y (Optional[pd.Series]): The target variable (not used in this processor).

        Returns:
        FeatureProcessor: The fitted processor.
        """
        # Create ratio features first if configured
        X_with_ratios = self._create_ratio_features(X)
        
        # Fit StandardScaler to the enhanced features
        self.scaler.fit(X_with_ratios)
        self._fitted = True
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Transform the input features based on the configuration.

        Parameters:
        X (pd.DataFrame): The input features to transform.

        Returns:
        pd.DataFrame: The transformed features.
        """
        if not self._fitted:
            raise ValueError("FeatureProcessor must be fitted before transform")
        
        # Create ratio features first if configured
        X_with_ratios = self._create_ratio_features(X)
        
        # Apply StandardScaler transformation
        X_scaled = self.scaler.transform(X_with_ratios)
        
        # Return as DataFrame with enhanced column names and index
        return pd.DataFrame(
            X_scaled,
            columns=X_with_ratios.columns,
            index=X.index
        )

    def fit_transform(self, X: pd.DataFrame, y: Optional[pd.Series] = None, **fit_params) -> pd.DataFrame:
        """
        Fit and transform the input features.

        Parameters:
        X (pd.DataFrame): The input features.
        y (Optional[pd.Series]): The target variable (not used in this processor).

        Returns:
        pd.DataFrame: The transformed features.
        """
        return self.fit(X, y).transform(X)

    def save(self, path: str) -> None:
        """
        Save the feature processor as an artifact

        Parameters:
        path (str): The file path to save the processor.
        """
        with open(path, 'wb') as f:
            pickle.dump(self, f)

    def load(self, path: str) -> 'FeatureProcessor':
        """
        Load the feature processor from a saved artifact.

        Parameters:
        path (str): The file path to load the processor from.

        Returns:
        FeatureProcessor: The loaded feature processor.
        """
        with open(path, 'rb') as f:
            return pickle.load(f)
