from typing import Optional, Tuple
import pandas as pd
import pickle

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import LabelEncoder

from iris_species_classification.config import DataConfig


class DataProcessor(BaseEstimator, TransformerMixin):
    def __init__(self, config: DataConfig):
        self.config: DataConfig = config
        self.label_encoder = LabelEncoder()
        self.feature_columns = config.feature_columns
        self.target_column = config.target_column
        self._fitted = False

    def fit(self, data: pd.DataFrame, y: Optional[pd.Series] = None) -> 'DataProcessor':
        """
        Fit the data processor to the data.

        Parameters:
        data (pd.DataFrame): The complete dataset with features and target.
        y (Optional[pd.Series]): Not used - target is extracted from data.

        Returns:
        DataProcessor: The fitted processor.
        """
        if self.target_column not in data.columns:
            raise ValueError(f"Target column '{self.target_column}' not found in data")
        
        # Fit label encoder on target column
        self.label_encoder.fit(data[self.target_column])
        self._fitted = True
        return self

    def transform(self, data: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Transform the input data by extracting features and encoding target.

        Parameters:
        data (pd.DataFrame): The complete dataset with features and target.

        Returns:
        Tuple[pd.DataFrame, pd.Series]: The transformed features and encoded target.
        """
        if not self._fitted:
            raise ValueError("DataProcessor must be fitted before transform")
        
        # Validate feature columns exist
        missing_features = [col for col in self.feature_columns if col not in data.columns]
        if missing_features:
            raise ValueError(f"Feature columns not found in data: {missing_features}")
        
        # Extract features
        X = data[self.feature_columns].copy()
        
        # Extract and encode target if present
        if self.target_column in data.columns:
            y = self.label_encoder.transform(data[self.target_column])
            y = pd.Series(y, index=data.index, name=self.target_column)
        else:
            # For prediction on new data without target
            y = None
            
        return X, y

    def transform_features_only(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Transform data to extract only features (for prediction).

        Parameters:
        data (pd.DataFrame): The dataset with features.

        Returns:
        pd.DataFrame: The extracted features.
        """
        # Validate feature columns exist
        missing_features = [col for col in self.feature_columns if col not in data.columns]
        if missing_features:
            raise ValueError(f"Feature columns not found in data: {missing_features}")
        
        return data[self.feature_columns].copy()

    def inverse_transform_target(self, y_encoded: pd.Series) -> pd.Series:
        """
        Inverse transform encoded target back to original labels.

        Parameters:
        y_encoded (pd.Series): Encoded target values.

        Returns:
        pd.Series: Original target labels.
        """
        if not self._fitted:
            raise ValueError("DataProcessor must be fitted before inverse_transform_target")
        
        return pd.Series(
            self.label_encoder.inverse_transform(y_encoded),
            index=y_encoded.index,
            name=self.target_column
        )

    def fit_transform(self, data: pd.DataFrame, y: Optional[pd.Series] = None, **fit_params) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Fit and transform the input data.

        Parameters:
        data (pd.DataFrame): The complete dataset with features and target.
        y (Optional[pd.Series]): Not used - target is extracted from data.

        Returns:
        Tuple[pd.DataFrame, pd.Series]: The transformed features and encoded target.
        """
        return self.fit(data, y).transform(data)

    def save(self, path: str) -> None:
        """
        Save the data processor as an artifact

        Parameters:
        path (str): The file path to save the processor.
        """
        with open(path, 'wb') as f:
            pickle.dump(self, f)

    def load(self, path: str) -> 'DataProcessor':
        """
        Load the data processor from a saved artifact.

        Parameters:
        path (str): The file path to load the processor from.

        Returns:
        DataProcessor: The loaded data processor.
        """
        with open(path, 'rb') as f:
            return pickle.load(f)
