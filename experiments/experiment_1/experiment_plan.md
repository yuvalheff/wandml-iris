# Iris Baseline Multi-Class Classification - Experiment Plan

## Experiment Overview

**Experiment Name:** Iris Baseline Multi-Class Classification  
**Iteration:** 1  
**Objective:** Establish a robust baseline for iris species classification using optimized Random Forest with standard preprocessing  

## Context and Rationale

This is the first iteration experiment based on comprehensive EDA analysis and algorithm exploration. The EDA revealed that:
- Petal measurements (length and width) are significantly more discriminative than sepal measurements
- Iris-setosa is clearly linearly separable from other species
- The dataset is perfectly balanced with 40 samples per species in the training set
- Iris-versicolor and Iris-virginica show some overlap requiring sophisticated classification

Algorithm exploration demonstrated that Random Forest with StandardScaler achieves perfect cross-validation performance while maintaining simplicity and interpretability.

## Data Preprocessing Steps

### 1. Data Loading
- **Input:** `data/train_set.csv` (120 samples, balanced distribution)
- **Action:** Load CSV file and verify data integrity
- **Expected:** 40 samples each of Iris-setosa, Iris-versicolor, Iris-virginica

### 2. Feature Extraction
- **Feature columns:** `SepalLengthCm`, `SepalWidthCm`, `PetalLengthCm`, `PetalWidthCm`
- **Target column:** `Species`
- **Exclude:** `Id` column (identifier, not predictive)
- **Action:** Create X_train from feature columns, y_train from Species column

### 3. Target Encoding
- **Method:** LabelEncoder from sklearn.preprocessing
- **Action:** Convert species names to numerical labels (0, 1, 2)
- **Classes:** Iris-setosa, Iris-versicolor, Iris-virginica

### 4. Feature Scaling
- **Method:** StandardScaler from sklearn.preprocessing
- **Rationale:** Ensures all features have similar scales, improves Random Forest performance
- **Action:** Fit StandardScaler on X_train, transform to create X_train_scaled

## Feature Engineering Steps

### Approach: No Additional Feature Engineering

**Rationale:** Exploration experiments revealed that the baseline 4 features achieve perfect cross-validation performance. Additional feature engineering approaches tested (ratio features, area features, polynomial features) did not improve performance and unnecessarily increased complexity.

**Features to Use:** All 4 original numerical features
- `SepalLengthCm` - Sepal length in centimeters
- `SepalWidthCm` - Sepal width in centimeters  
- `PetalLengthCm` - Petal length in centimeters (highly discriminative)
- `PetalWidthCm` - Petal width in centimeters (most discriminative)

**Scaling:** StandardScaler applied to normalize feature distributions

## Model Selection Steps

### Primary Algorithm: Random Forest Classifier

**Rationale:** 
- Achieved perfect cross-validation AUC (1.0000) with StandardScaler
- Demonstrated robust performance on test set (0.9867 macro AUC)
- Provides interpretable feature importance analysis
- Handles non-linear relationships and feature interactions naturally
- Robust to outliers and overfitting with proper hyperparameters

### Hyperparameters (Optimized via GridSearchCV)
- **n_estimators:** 200 (optimal balance of performance and training time)
- **max_depth:** 3 (prevents overfitting while capturing key patterns)
- **min_samples_split:** 2 (allows fine-grained splits for class separation)
- **random_state:** 42 (reproducibility)

### Hyperparameter Tuning Process
1. **Method:** GridSearchCV with 5-fold stratified cross-validation
2. **Scoring:** roc_auc_ovr_weighted (matches target metric)
3. **Parameter Grid:**
   - n_estimators: [50, 100, 200]
   - max_depth: [3, 5, 7, None]
   - min_samples_split: [2, 5]

## Evaluation Strategy

### Primary Metric
**Macro-averaged AUC:** `roc_auc_score` with `multi_class='ovr'` and `average='macro'`

### Cross-Validation Approach
- **Method:** StratifiedKFold (n_splits=5, shuffle=True, random_state=42)
- **Rationale:** Maintains class balance across folds, provides robust performance estimate

### Test Set Evaluation
- **Input:** `data/test_set.csv` (30 samples, 10 per species)
- **Metrics:**
  - Macro AUC (primary metric)
  - Accuracy, Precision, Recall, F1-score (macro-averaged)
  - Per-class metrics via classification_report
  - Confusion matrix for error analysis

### Diagnostic Analyses

#### 1. Feature Importance Analysis
- **Purpose:** Understand which measurements drive classification decisions
- **Method:** Extract `feature_importances_` from trained Random Forest
- **Expected:** Petal measurements should rank highest (confirmed by EDA)

#### 2. Per-Class Performance Analysis  
- **Purpose:** Identify class-specific strengths and weaknesses
- **Method:** Analyze precision, recall, F1-score for each species
- **Expected:** Iris-setosa perfect classification, some confusion between versicolor/virginica

#### 3. Cross-Validation Stability Assessment
- **Purpose:** Evaluate model robustness and consistency
- **Method:** Report CV score distribution (mean ± std)
- **Expected:** Low variance indicating stable performance

#### 4. Probability Calibration Assessment
- **Purpose:** Ensure well-calibrated prediction confidence
- **Method:** Analyze predicted probability distributions per class
- **Output:** Confidence level analysis for each species prediction

## Expected Outputs

### Model Artifacts
- `trained_model.pkl` - Serialized Random Forest with fitted StandardScaler
- `predictions.csv` - Test predictions with class probabilities
- `feature_importance.csv` - Ranked feature importance scores

### Evaluation Reports
- `evaluation_report.json` - Comprehensive metrics and diagnostic results
- `cv_results.json` - Detailed cross-validation scores and statistics
- `confusion_matrix.png` - Visual confusion matrix for test set

### Key Performance Targets
- **Cross-validation macro AUC:** ≥0.99 (based on exploration results)
- **Test set macro AUC:** ≥0.95 (accounting for potential overfitting)
- **Test set accuracy:** ≥0.90 (high accuracy expected for this dataset)
- **Per-class F1 scores:** ≥0.80 for all species

## Success Criteria

### Primary Success Criterion
- Test set macro AUC ≥ 0.95

### Secondary Success Criteria
- Cross-validation macro AUC ≥ 0.99
- Test set accuracy ≥ 0.90  
- All species classes achieve F1-score ≥ 0.80
- Feature importance ranking aligns with EDA insights (petal > sepal)

## Implementation Notes

1. **Data Loading:** Ensure proper handling of CSV format and data types
2. **Preprocessing Pipeline:** Apply same scaling transformation to test set using fitted scaler
3. **Model Training:** Use stratified sampling to maintain class balance
4. **Evaluation:** Calculate macro AUC carefully using sklearn's multi-class implementation
5. **Reproducibility:** Set random_state consistently across all random components

This experiment plan provides a solid foundation for iris species classification with clear success criteria and comprehensive evaluation strategy.