import pandas as pd
import numpy as np
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import roc_auc_score, classification_report
from sklearn.preprocessing import LabelEncoder
import warnings
warnings.filterwarnings('ignore')

# Load data
train_path = '/Users/yuvalheffetz/ds-agent-projects/session_8f5d6987-98d5-4c7f-84cd-6fe7a7f21976/data/train_set.csv'
test_path = '/Users/yuvalheffetz/ds-agent-projects/session_8f5d6987-98d5-4c7f-84cd-6fe7a7f21976/data/test_set.csv'

train_df = pd.read_csv(train_path)
test_df = pd.read_csv(test_path)

print("=== DATA OVERVIEW ===")
print(f"Train shape: {train_df.shape}")
print(f"Test shape: {test_df.shape}")
print(f"Target distribution in train:")
print(train_df['Species'].value_counts())

# Prepare features
feature_cols = ['SepalLengthCm', 'SepalWidthCm', 'PetalLengthCm', 'PetalWidthCm']
X_train = train_df[feature_cols]
y_train = train_df['Species']

X_test = test_df[feature_cols]
y_test = test_df['Species']

# Convert target to numeric for roc_auc
le = LabelEncoder()
y_train_encoded = le.fit_transform(y_train)
y_test_encoded = le.transform(y_test)

print("\n=== FEATURE STATISTICS ===")
print(X_train.describe())

def evaluate_model_multiclass(model, X, y, model_name, cv_folds=5):
    """Evaluate multiclass model using stratified CV and macro AUC"""
    skf = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42)
    
    # Cross-validation scores
    cv_scores = cross_val_score(model, X, y, cv=skf, scoring='roc_auc_ovr_weighted')
    
    return {
        'model': model_name,
        'mean_cv_auc': cv_scores.mean(),
        'std_cv_auc': cv_scores.std(),
        'cv_scores': cv_scores
    }

# Test different models
models = {
    'Logistic Regression': LogisticRegression(random_state=42, max_iter=1000),
    'Random Forest': RandomForestClassifier(random_state=42, n_estimators=100),
    'SVM': SVC(random_state=42, probability=True),
    'Gaussian NB': GaussianNB(),
    'KNN': KNeighborsClassifier(n_neighbors=5),
    'Decision Tree': DecisionTreeClassifier(random_state=42, max_depth=5)
}

print("\n=== MODEL COMPARISON (No Scaling) ===")
results = []
for name, model in models.items():
    result = evaluate_model_multiclass(model, X_train, y_train_encoded, name)
    results.append(result)
    print(f"{name}: {result['mean_cv_auc']:.4f} (+/- {result['std_cv_auc']:.4f})")

# Test with scaling
print("\n=== MODEL COMPARISON (With StandardScaler) ===")
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)

results_scaled = []
for name, model in models.items():
    result = evaluate_model_multiclass(model, X_train_scaled, y_train_encoded, name + "_scaled")
    results_scaled.append(result)
    print(f"{name} (scaled): {result['mean_cv_auc']:.4f} (+/- {result['std_cv_auc']:.4f})")

# Test with MinMax scaling
print("\n=== MODEL COMPARISON (With MinMaxScaler) ===")
minmax_scaler = MinMaxScaler()
X_train_minmax = minmax_scaler.fit_transform(X_train)

results_minmax = []
for name, model in models.items():
    result = evaluate_model_multiclass(model, X_train_minmax, y_train_encoded, name + "_minmax")
    results_minmax.append(result)
    print(f"{name} (minmax): {result['mean_cv_auc']:.4f} (+/- {result['std_cv_auc']:.4f})")

# Feature importance analysis with Random Forest
print("\n=== FEATURE IMPORTANCE ANALYSIS ===")
rf = RandomForestClassifier(random_state=42, n_estimators=100)
rf.fit(X_train, y_train_encoded)

feature_importance = pd.DataFrame({
    'feature': feature_cols,
    'importance': rf.feature_importances_
}).sort_values('importance', ascending=False)

print("Feature importance (Random Forest):")
print(feature_importance)

# Test best performing model on test set
print("\n=== BEST MODEL EVALUATION ON TEST SET ===")
best_models = []
for results_set in [results, results_scaled, results_minmax]:
    best_model = max(results_set, key=lambda x: x['mean_cv_auc'])
    best_models.append(best_model)

overall_best = max(best_models, key=lambda x: x['mean_cv_auc'])
print(f"Overall best model: {overall_best['model']} with CV AUC: {overall_best['mean_cv_auc']:.4f}")

# Final model training and test evaluation
if 'scaled' in overall_best['model']:
    X_train_final = X_train_scaled
    X_test_final = scaler.transform(X_test)
    model_name_clean = overall_best['model'].replace('_scaled', '')
elif 'minmax' in overall_best['model']:
    X_train_final = X_train_minmax
    X_test_final = minmax_scaler.transform(X_test)
    model_name_clean = overall_best['model'].replace('_minmax', '')
else:
    X_train_final = X_train
    X_test_final = X_test
    model_name_clean = overall_best['model']

final_model = models[model_name_clean]
final_model.fit(X_train_final, y_train_encoded)
y_pred_proba = final_model.predict_proba(X_test_final)

# Calculate macro AUC
test_auc = roc_auc_score(y_test_encoded, y_pred_proba, multi_class='ovr', average='macro')
print(f"Final test set macro AUC: {test_auc:.4f}")

# Classification report
y_pred = final_model.predict(X_test_final)
print("\nClassification Report:")
print(classification_report(y_test_encoded, y_pred, target_names=le.classes_))

print("\n=== HYPERPARAMETER TESTING ===")
# Test different hyperparameters for top models
from sklearn.model_selection import GridSearchCV

if model_name_clean == 'Random Forest':
    rf_params = {
        'n_estimators': [50, 100, 200],
        'max_depth': [3, 5, 7, None],
        'min_samples_split': [2, 5]
    }
    grid_search = GridSearchCV(RandomForestClassifier(random_state=42), 
                              rf_params, cv=5, scoring='roc_auc_ovr_weighted')
    grid_search.fit(X_train_final, y_train_encoded)
    print(f"Best RF params: {grid_search.best_params_}")
    print(f"Best RF score: {grid_search.best_score_:.4f}")

elif model_name_clean == 'SVM':
    svm_params = {
        'C': [0.1, 1, 10],
        'gamma': ['scale', 'auto', 0.01, 0.1],
        'kernel': ['rbf', 'linear']
    }
    grid_search = GridSearchCV(SVC(random_state=42, probability=True), 
                              svm_params, cv=5, scoring='roc_auc_ovr_weighted')
    grid_search.fit(X_train_final, y_train_encoded)
    print(f"Best SVM params: {grid_search.best_params_}")
    print(f"Best SVM score: {grid_search.best_score_:.4f}")

elif model_name_clean == 'Logistic Regression':
    lr_params = {
        'C': [0.1, 1, 10, 100],
        'penalty': ['l1', 'l2'],
        'solver': ['liblinear', 'saga']
    }
    grid_search = GridSearchCV(LogisticRegression(random_state=42, max_iter=1000), 
                              lr_params, cv=5, scoring='roc_auc_ovr_weighted')
    grid_search.fit(X_train_final, y_train_encoded)
    print(f"Best LR params: {grid_search.best_params_}")
    print(f"Best LR score: {grid_search.best_score_:.4f}")