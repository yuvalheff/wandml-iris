# Iris Species Classification - EDA Report

## Dataset Overview
The Iris dataset contains 120 training samples with 4 numerical features and 1 categorical target variable. This is R.A. Fisher's famous 1936 dataset for taxonomic classification, featuring three balanced classes of iris flower species with 40 samples each.

## Features Analysis

### Numerical Features
- **SepalLengthCm**: Length of the sepal in centimeters (continuous, moderate discriminative power)
- **SepalWidthCm**: Width of the sepal in centimeters (continuous, lower discriminative power) 
- **PetalLengthCm**: Length of the petal in centimeters (continuous, highly discriminative)
- **PetalWidthCm**: Width of the petal in centimeters (continuous, most discriminative)

### Target Variable
- **Species**: Three iris flower species - Iris-setosa, Iris-versicolor, Iris-virginica (perfectly balanced)

## Key Findings

### Feature Distribution Analysis by Species
**Analysis**: Analyzed the distribution of all four numerical features across the three iris species using violin plots.

**Key Insights**:
1. **Petal measurements** (length and width) show much clearer separation between species compared to sepal measurements, making them more discriminative features for classification
2. **Iris-setosa** is clearly linearly separable from the other two species across all features, especially petal measurements, with non-overlapping distributions  
3. **Iris-versicolor and Iris-virginica** show some overlap in their feature distributions, particularly in sepal measurements, making them harder to distinguish from each other

## Summary
The Iris dataset analysis reveals a well-structured classification problem with clear patterns. The dataset contains 120 balanced training samples across three species. Feature analysis shows that petal measurements (length and width) are significantly more discriminative than sepal measurements for species classification. Iris-setosa exhibits the most distinct feature distributions and appears linearly separable from the other species, while Iris-versicolor and Iris-virginica show overlapping distributions that will require more sophisticated classification approaches. The data is clean with no missing values and demonstrates the classic machine learning challenge where one class is easily separable while two classes require more nuanced differentiation.

## Generated Plots
- `feature_distribution_by_species.html`: Interactive violin plots showing feature distributions by species