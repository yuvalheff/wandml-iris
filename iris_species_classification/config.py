from dataclasses import dataclass
from typing import Any, Dict, List, Optional
import yaml


class ConfigParsingFailed(Exception):
    pass


@dataclass
class DataConfig:
    dataset_name: str
    feature_columns: List[str]
    target_column: str


@dataclass 
class RatioFeatureConfig:
    name: str
    numerator: str
    denominator: str

@dataclass
class FeaturesConfig:
    scaling_method: str = "StandardScaler"
    create_ratio_features: bool = False
    ratio_features: List[RatioFeatureConfig] = None
    
    def __post_init__(self):
        if self.ratio_features is None:
            self.ratio_features = []
        elif isinstance(self.ratio_features, list) and self.ratio_features:
            # Convert dict to RatioFeatureConfig objects if needed
            if isinstance(self.ratio_features[0], dict):
                self.ratio_features = [RatioFeatureConfig(**rf) for rf in self.ratio_features]


@dataclass
class ModelEvalConfig:
    primary_metric: str
    cv_folds: int
    cv_method: str
    test_size: float
    random_state: int


@dataclass
class ModelConfig:
    model_type: str
    model_params: Dict[str, Any]


@dataclass
class Config:
    data_prep: DataConfig
    feature_prep: FeaturesConfig
    model_evaluation: ModelEvalConfig
    model: ModelConfig

    @staticmethod
    def from_yaml(config_file: str):
        with open(config_file, 'r', encoding='utf-8') as stream:
            try:
                config_data = yaml.safe_load(stream)
                return Config(
                    data_prep=DataConfig(**config_data['data_prep']),
                    feature_prep=FeaturesConfig(**config_data['feature_prep']),
                    model_evaluation=ModelEvalConfig(**config_data['model_evaluation']),
                    model=ModelConfig(**config_data['model'])
                )
            except (yaml.YAMLError, OSError) as e:
                raise ConfigParsingFailed from e