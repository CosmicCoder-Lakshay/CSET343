# Lab 4 – Logistic Regression for Breast Cancer Diagnosis

## Course
- Course Code: CSET343
- Course Name: AI in Healthcare
- Semester: VII
- Year: 4th Year

## Objective
To implement Logistic Regression for binary classification using the Breast Cancer Wisconsin (Diagnostic) dataset.

## Dataset
Breast Cancer Wisconsin (Diagnostic) Dataset.

## Tasks Performed
1. Loaded and explored the dataset.
2. Calculated summary statistics.
3. Checked missing values and duplicate records.
4. Detected outliers using box plots.
5. Visualized class distribution and feature correlations.
6. Encoded diagnosis as binary:
   - B = 0 (Benign)
   - M = 1 (Malignant)
7. Split data into 80% training and 20% testing using stratified sampling.
8. Standardized features using StandardScaler.
9. Trained a Logistic Regression model.
10. Evaluated the model using:
   - Accuracy
   - Precision
   - Recall
   - F1-score
   - Confusion Matrix
   - ROC Curve
   - AUC

## Results
- Accuracy: 96.49%
- AUC Score: 0.996

## Conclusion
The Logistic Regression model achieved high classification performance on the Breast Cancer Wisconsin dataset. The results show that the model can effectively distinguish between benign and malignant cases.