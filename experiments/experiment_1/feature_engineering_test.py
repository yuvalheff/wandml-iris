import pandas as pd
import numpy as np
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder, PolynomialFeatures
from sklearn.decomposition import PCA
import warnings
warnings.filterwarnings('ignore')

# Load data
train_path = '/Users/yuvalheffetz/ds-agent-projects/session_8f5d6987-98d5-4c7f-84cd-6fe7a7f21976/data/train_set.csv'

train_df = pd.read_csv(train_path)

# Prepare features
feature_cols = ['SepalLengthCm', 'SepalWidthCm', 'PetalLengthCm', 'PetalWidthCm']
X_train = train_df[feature_cols]
y_train = train_df['Species']

# Convert target to numeric
le = LabelEncoder()
y_train_encoded = le.fit_transform(y_train)

def evaluate_features(X, y, feature_name, model=None):
    """Evaluate feature set using stratified CV"""
    if model is None:
        model = RandomForestClassifier(random_state=42, n_estimators=200, max_depth=3, min_samples_split=2)
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cv_scores = cross_val_score(model, X_scaled, y, cv=skf, scoring='roc_auc_ovr_weighted')
    
    return {
        'feature_set': feature_name,
        'mean_auc': cv_scores.mean(),
        'std_auc': cv_scores.std(),
        'n_features': X.shape[1]
    }

print("=== FEATURE ENGINEERING EXPERIMENTS ===")

# Baseline
baseline_result = evaluate_features(X_train, y_train_encoded, "Baseline (4 features)")
print(f"Baseline: {baseline_result['mean_auc']:.4f} (+/- {baseline_result['std_auc']:.4f}) - {baseline_result['n_features']} features")

# 1. Ratio features
X_ratio = X_train.copy()
X_ratio['SepalRatio'] = X_train['SepalLengthCm'] / X_train['SepalWidthCm']
X_ratio['PetalRatio'] = X_train['PetalLengthCm'] / X_train['PetalWidthCm']
X_ratio['Length_Width_Ratio'] = (X_train['SepalLengthCm'] + X_train['PetalLengthCm']) / (X_train['SepalWidthCm'] + X_train['PetalWidthCm'])

ratio_result = evaluate_features(X_ratio, y_train_encoded, "With Ratio Features")
print(f"Ratio features: {ratio_result['mean_auc']:.4f} (+/- {ratio_result['std_auc']:.4f}) - {ratio_result['n_features']} features")

# 2. Area features
X_area = X_train.copy()
X_area['SepalArea'] = X_train['SepalLengthCm'] * X_train['SepalWidthCm']
X_area['PetalArea'] = X_train['PetalLengthCm'] * X_train['PetalWidthCm']
X_area['TotalArea'] = X_area['SepalArea'] + X_area['PetalArea']

area_result = evaluate_features(X_area, y_train_encoded, "With Area Features")
print(f"Area features: {area_result['mean_auc']:.4f} (+/- {area_result['std_auc']:.4f}) - {area_result['n_features']} features")

# 3. Polynomial features (degree 2)
poly = PolynomialFeatures(degree=2, include_bias=False, interaction_only=False)
X_poly = poly.fit_transform(X_train)

poly_result = evaluate_features(X_poly, y_train_encoded, "Polynomial Features (degree 2)")
print(f"Polynomial features: {poly_result['mean_auc']:.4f} (+/- {poly_result['std_auc']:.4f}) - {poly_result['n_features']} features")

# 4. Interaction features only
poly_int = PolynomialFeatures(degree=2, include_bias=False, interaction_only=True)
X_poly_int = poly_int.fit_transform(X_train)

poly_int_result = evaluate_features(X_poly_int, y_train_encoded, "Interaction Features Only")
print(f"Interaction features: {poly_int_result['mean_auc']:.4f} (+/- {poly_int_result['std_auc']:.4f}) - {poly_int_result['n_features']} features")

# 5. PCA features
for n_components in [2, 3, 4]:
    pca = PCA(n_components=n_components)
    X_pca = pca.fit_transform(X_train)
    
    pca_result = evaluate_features(X_pca, y_train_encoded, f"PCA ({n_components} components)")
    print(f"PCA {n_components} components: {pca_result['mean_auc']:.4f} (+/- {pca_result['std_auc']:.4f}) - {pca_result['n_features']} features")

# 6. Combined engineered features
X_combined = X_train.copy()
# Add ratio features
X_combined['SepalRatio'] = X_train['SepalLengthCm'] / X_train['SepalWidthCm']
X_combined['PetalRatio'] = X_train['PetalLengthCm'] / X_train['PetalWidthCm']
# Add area features
X_combined['SepalArea'] = X_train['SepalLengthCm'] * X_train['SepalWidthCm']
X_combined['PetalArea'] = X_train['PetalLengthCm'] * X_train['PetalWidthCm']

combined_result = evaluate_features(X_combined, y_train_encoded, "Combined Engineered Features")
print(f"Combined features: {combined_result['mean_auc']:.4f} (+/- {combined_result['std_auc']:.4f}) - {combined_result['n_features']} features")

# 7. Only petal features (based on feature importance)
X_petal = X_train[['PetalLengthCm', 'PetalWidthCm']]

petal_result = evaluate_features(X_petal, y_train_encoded, "Petal Features Only")
print(f"Petal only: {petal_result['mean_auc']:.4f} (+/- {petal_result['std_auc']:.4f}) - {petal_result['n_features']} features")

print("\n=== SUMMARY OF FEATURE ENGINEERING ===")
results = [baseline_result, ratio_result, area_result, poly_result, poly_int_result, combined_result, petal_result]
results_sorted = sorted(results, key=lambda x: x['mean_auc'], reverse=True)

for i, result in enumerate(results_sorted, 1):
    print(f"{i}. {result['feature_set']}: {result['mean_auc']:.4f} (+/- {result['std_auc']:.4f}) - {result['n_features']} features")

print(f"\nBest feature engineering approach: {results_sorted[0]['feature_set']}")