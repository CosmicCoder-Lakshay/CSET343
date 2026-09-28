# Lab 7 - Hyperparameter Tuning of Neural Network

## Course
AI in Healthcare - CSET343

## Objective

The objective of this lab is to develop a neural network model for diabetes
prediction using the Pima Indians Diabetes Dataset and perform hyperparameter
tuning using GridSearchCV.

## Dataset

The Pima Indians Diabetes Dataset contains medical diagnostic measurements
used to predict whether a patient has diabetes.

### Target Variable

- `Outcome = 0` → No Diabetes
- `Outcome = 1` → Diabetes

## Methodology

The following steps are performed:

1. Load the diabetes dataset.
2. Separate input features and target variable.
3. Handle invalid zero values in clinical features.
4. Replace missing values using median imputation.
5. Standardize numerical features using StandardScaler.
6. Split the dataset into training and testing sets.
7. Build an Artificial Neural Network using TensorFlow/Keras.
8. Perform hyperparameter tuning using GridSearchCV.
9. Evaluate the final model on the test dataset.

## Hyperparameters Tuned

The following parameters are tuned:

- Batch Size
- Number of Epochs
- Learning Rate
- Weight Initialization
- Activation Function
- Dropout Rate
- Number of Hidden Neurons

## Technologies Used

- Python
- NumPy
- Pandas
- Scikit-learn
- TensorFlow
- Keras
- SciKeras

## Files

```text
Lab7/
├── Lab7.py
├── diabetes.csv
├── LabAssignment_7.pdf
├── README.md
└── requirements.txt