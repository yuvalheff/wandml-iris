import pandas as pd
import numpy as np
import json
import pickle
from pathlib import Path
import os
from typing import Dict, Any
import mlflow
import mlflow.sklearn
import sklearn

from iris_species_classification.pipeline.feature_preprocessing import FeatureProcessor
from iris_species_classification.pipeline.data_preprocessing import DataProcessor
from iris_species_classification.pipeline.model import ModelWrapper
from iris_species_classification.config import Config
from iris_species_classification.model_pipeline import ModelPipeline
from experiment_scripts.evaluation import ModelEvaluator

DEFAULT_CONFIG = str(Path(__file__).parent / 'config.yaml')


class Experiment:
    def __init__(self):
        self._config = Config.from_yaml(DEFAULT_CONFIG)

    def run(self, train_dataset_path: str, test_dataset_path: str, output_dir: str, seed: int = 42) -> Dict[str, Any]:
        """
        Run the complete Iris species classification experiment.
        
        Parameters:
        train_dataset_path: Path to training dataset CSV
        test_dataset_path: Path to test dataset CSV  
        output_dir: Output directory for artifacts and plots
        seed: Random seed for reproducibility
        
        Returns:
        Dict containing experiment results and metadata
        """
        print(f"🚀 Starting Iris Species Classification Experiment")
        print(f"📊 Train data: {train_dataset_path}")
        print(f"🧪 Test data: {test_dataset_path}")
        print(f"📁 Output dir: {output_dir}")
        print(f"🎲 Seed: {seed}")
        
        # Set random seeds for reproducibility
        np.random.seed(seed)
        
        # Create output directories
        os.makedirs(output_dir, exist_ok=True)
        plots_dir = os.path.join(output_dir, "output", "plots")
        artifacts_dir = os.path.join(output_dir, "output", "model_artifacts")
        general_artifacts_dir = os.path.join(output_dir, "output", "general_artifacts")
        
        os.makedirs(plots_dir, exist_ok=True)
        os.makedirs(artifacts_dir, exist_ok=True)
        os.makedirs(general_artifacts_dir, exist_ok=True)
        
        # Load and preprocess data
        print("📈 Loading and preprocessing data...")
        train_data = pd.read_csv(train_dataset_path)
        test_data = pd.read_csv(test_dataset_path)
        
        # Initialize processors
        data_processor = DataProcessor(self._config.data_prep)
        feature_processor = FeatureProcessor(self._config.feature_prep)
        model = ModelWrapper(self._config.model)
        evaluator = ModelEvaluator(self._config.model_evaluation)
        
        # Fit data processor and transform training data
        data_processor.fit(train_data)
        X_train, y_train = data_processor.transform(train_data)
        
        # Fit feature processor and transform training features
        feature_processor.fit(X_train)
        X_train_scaled = feature_processor.transform(X_train)
        
        # Train model
        print("🔧 Training model...")
        model.fit(X_train_scaled, y_train)
        
        # Cross-validation evaluation
        print("📊 Performing cross-validation...")
        cv_results = evaluator.cross_validate_model(model, X_train_scaled, y_train)
        print(f"✅ Cross-validation AUC: {cv_results['cv_mean']:.4f} ± {cv_results['cv_std']:.4f}")
        
        # Transform test data and evaluate
        print("🎯 Evaluating on test set...")
        X_test, y_test = data_processor.transform(test_data)
        X_test_scaled = feature_processor.transform(X_test)
        
        # Get class labels for interpretation
        class_labels = data_processor.label_encoder.classes_.tolist()
        
        # Comprehensive test evaluation with plots
        test_results = evaluator.evaluate_model(
            model, X_test_scaled, y_test, class_labels, plots_dir
        )
        
        print(f"🎯 Test Results:")
        print(f"   • Macro AUC: {test_results['macro_auc']:.4f}")
        print(f"   • Accuracy: {test_results['accuracy']:.4f}")
        print(f"   • F1 Macro: {test_results['f1_macro']:.4f}")
        
        # Save individual artifacts
        print("💾 Saving model artifacts...")
        data_processor.save(os.path.join(artifacts_dir, "data_processor.pkl"))
        feature_processor.save(os.path.join(artifacts_dir, "feature_processor.pkl"))
        model.save(os.path.join(artifacts_dir, "trained_model.pkl"))
        
        # Create and test ModelPipeline
        print("🔄 Creating ModelPipeline...")
        pipeline = ModelPipeline(data_processor, feature_processor, model)
        
        # Test pipeline with sample data
        sample_input = X_test.iloc[:1].copy()  # Use first test sample
        sample_prediction = pipeline.predict(sample_input)
        sample_proba = pipeline.predict_proba(sample_input)
        
        print(f"🧪 Pipeline test successful - Sample prediction: {sample_prediction[0]}")
        
        # Create MLflow model
        print("🏗️ Creating MLflow model...")
        
        # Define the standard local output path
        mlflow_model_path = os.path.join(artifacts_dir, "mlflow_model")
        relative_path_for_return = "output/model_artifacts/mlflow_model/"
        
        # Clean up existing MLflow model directory if it exists
        if os.path.exists(mlflow_model_path):
            import shutil
            shutil.rmtree(mlflow_model_path)
        
        # Create model signature
        signature = mlflow.models.infer_signature(sample_input, pipeline.predict(sample_input))
        
        # 1. Always save the model to the local path for harness validation
        print(f"💾 Saving model to local disk for harness: {mlflow_model_path}")
        mlflow.sklearn.save_model(
            pipeline,
            path=mlflow_model_path,
            signature=signature
        )
        
        # 2. If an MLflow run ID is provided, reconnect and log the model as an artifact
        active_run_id = "e76c52d55a494937987499eb13d60f93"
        logged_model_uri = None  # Initialize to None
        
        if active_run_id and active_run_id != 'None' and active_run_id.strip():
            print(f"✅ Active MLflow run ID '{active_run_id}' detected. Reconnecting to log model as an artifact.")
            with mlflow.start_run(run_id=active_run_id):
                logged_model_info = mlflow.sklearn.log_model(
                    pipeline,
                    artifact_path="model",  # Use a standard artifact path
                    code_paths=["iris_species_classification"], # Bundle the custom code
                    signature=signature
                )
                logged_model_uri = logged_model_info.model_uri
        else:
            print("ℹ️ No active MLflow run ID provided. Skipping model logging.")
        
        # Save comprehensive results
        comprehensive_results = {
            'experiment_config': {
                'train_dataset_path': train_dataset_path,
                'test_dataset_path': test_dataset_path,
                'seed': seed,
                'model_type': self._config.model.model_type,
                'model_params': self._config.model.model_params
            },
            'cross_validation': cv_results,
            'test_evaluation': test_results,
            'pipeline_info': {
                'feature_columns': pipeline.get_feature_names(),
                'target_classes': pipeline.get_target_classes()
            }
        }
        
        # Save results to general artifacts
        with open(os.path.join(general_artifacts_dir, "experiment_results.json"), 'w') as f:
            json.dump(comprehensive_results, f, indent=2, default=str)
        
        # Prepare model artifacts list
        model_artifacts = [
            "data_processor.pkl",
            "feature_processor.pkl", 
            "trained_model.pkl",
            "mlflow_model/"
        ]
        
        # Prepare return dictionary with mandatory format
        # Map config metric name to test result key
        metric_mapping = {
            'macro_auc': 'macro_auc',
            'accuracy': 'accuracy',
            'f1_macro': 'f1_macro'
        }
        primary_metric_key = metric_mapping.get(self._config.model_evaluation.primary_metric, 'macro_auc')
        primary_metric_value = test_results[primary_metric_key]
        
        return {
            "metric_name": self._config.model_evaluation.primary_metric,
            "metric_value": float(primary_metric_value),
            "model_artifacts": model_artifacts,
            "mlflow_model_info": {
                "model_path": relative_path_for_return,
                "logged_model_uri": logged_model_uri,
                "model_type": "sklearn",
                "task_type": "classification", 
                "signature": {
                    "inputs": signature.inputs.to_dict() if signature and signature.inputs else None,
                    "outputs": signature.outputs.to_dict() if signature and signature.outputs else None
                },
                "input_example": sample_input.iloc[0].to_dict(),
                "framework_version": sklearn.__version__
            }
        }