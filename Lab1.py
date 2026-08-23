import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

print("=" * 60)
print("AI IN HEALTHCARE - LAB EXPERIMENT 1")
print("=" * 60)

# Cleveland Heart Disease Dataset
file_path = "heart+disease/processed.cleveland.data"

columns = [
    "age",
    "sex",
    "cp",
    "trestbps",
    "chol",
    "fbs",
    "restecg",
    "thalach",
    "exang",
    "oldpeak",
    "slope",
    "ca",
    "thal",
    "target"
]

heart_df = pd.read_csv(
    file_path,
    names=columns,
    na_values="?"
)

print("\nHeart Disease Dataset loaded successfully!")
print("Dataset shape:", heart_df.shape)

print("\nFirst 5 records:")
print(heart_df.head())
# ============================================================
# DATA INSPECTION
# ============================================================

print("\nDataset Information:")
heart_df.info()

print("\nMissing Values:")
print(heart_df.isnull().sum())

print("\nStatistical Summary:")
print(heart_df.describe())

# ============================================================
# MISSING VALUE HANDLING
# ============================================================

print("\nMissing Values Before Handling:")
print(heart_df.isnull().sum())

# Fill missing values using median
heart_df["ca"] = heart_df["ca"].fillna(heart_df["ca"].median())
heart_df["thal"] = heart_df["thal"].fillna(heart_df["thal"].median())

print("\nMissing Values After Handling:")
print(heart_df.isnull().sum())


# ============================================================
# TARGET PREPROCESSING
# ============================================================

heart_df["target_binary"] = (heart_df["target"] > 0).astype(int)

print("\nTarget Distribution:")
print(heart_df["target_binary"].value_counts())

print("\nTarget Meaning:")
print("0 = No Heart Disease")
print("1 = Heart Disease")
# ============================================================
# DATA VISUALIZATION
# ============================================================

plt.figure(figsize=(8, 5))

plt.hist(
    heart_df[heart_df["target_binary"] == 0]["age"],
    bins=15,
    alpha=0.6,
    label="No Heart Disease"
)

plt.hist(
    heart_df[heart_df["target_binary"] == 1]["age"],
    bins=15,
    alpha=0.6,
    label="Heart Disease"
)

plt.xlabel("Age")
plt.ylabel("Number of Patients")
plt.title("Age Distribution by Heart Disease Status")
plt.legend()

plt.show()
# ============================================================
# MACHINE LEARNING PREPARATION
# ============================================================

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    ConfusionMatrixDisplay,
    roc_auc_score,
    RocCurveDisplay
)

# ============================================================
# MACHINE LEARNING PREPARATION
# ============================================================

# Features: remove both original target and binary target
X = heart_df.drop(columns=["target", "target_binary"]).copy()

# Target
y = heart_df["target_binary"].copy()

print("\nFeature columns:")
print(X.columns.tolist())

print("\nFeatures shape:", X.shape)
print("Target shape:", y.shape)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print("\nTraining samples:", len(X_train))
print("Testing samples:", len(X_test))

# Make sure all features are numeric
X = X.apply(pd.to_numeric)

print("\nFeatures shape:", X.shape)
print("Target shape:", y.shape)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print("\nTraining samples:", len(X_train))
print("Testing samples:", len(X_test))
# ============================================================
# FEATURE SCALING
# ============================================================

scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print("\nFeature scaling completed!")
print("Scaled training data shape:", X_train_scaled.shape)
print("Scaled testing data shape:", X_test_scaled.shape)
# ============================================================
# LOGISTIC REGRESSION MODEL
# ============================================================

model = LogisticRegression(max_iter=1000)

model.fit(X_train_scaled, y_train)

print("\nLogistic Regression model trained successfully!")
# ============================================================
# MODEL PREDICTION AND EVALUATION
# ============================================================

# Predict classes
y_pred = model.predict(X_test_scaled)

# Predict probability of heart disease
y_prob = model.predict_proba(X_test_scaled)[:, 1]

# Accuracy
accuracy = accuracy_score(y_test, y_pred)

# ROC-AUC
auc = roc_auc_score(y_test, y_prob)

print("\nModel Performance")
print("-" * 40)
print("Accuracy :", round(accuracy, 4))
print("ROC-AUC  :", round(auc, 4))
# ============================================================
# CONFUSION MATRIX
# ============================================================

cm = confusion_matrix(y_test, y_pred)

disp = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=["No Disease", "Disease"]
)

disp.plot()

plt.title("Confusion Matrix - Heart Disease Prediction")
plt.show()
# ============================================================
# ROC CURVE
# ============================================================

RocCurveDisplay.from_predictions(
    y_test,
    y_prob
)

plt.title("ROC Curve - Heart Disease Prediction")
plt.grid(True)

plt.show()


# ============================================================
# PART B - SIGNAL DATA
# MIT-BIH ARRHYTHMIA DATABASE
# ============================================================

import wfdb

record_name = "100"

# Read ECG signal
record = wfdb.rdrecord(
    record_name,
    pn_dir="mitdb"
)

# Read ECG annotations
annotation = wfdb.rdann(
    record_name,
    "atr",
    pn_dir="mitdb"
)

print("\nECG Record Loaded Successfully!")

print("Record Name:", record_name)
print("Sampling Frequency:", record.fs, "Hz")
print("Number of Samples:", record.sig_len)
print("Number of Channels:", record.n_sig)
print("Signal Shape:", record.p_signal.shape)
# ============================================================
# ECG SIGNAL VISUALIZATION
# ============================================================

# Display first 10 seconds of ECG
duration = 10

samples = int(duration * record.fs)

# Select first ECG channel
signal = record.p_signal[:samples, 0]

# Create time axis
time = np.arange(len(signal)) / record.fs

plt.figure(figsize=(15, 5))

plt.plot(time, signal)

plt.xlabel("Time (seconds)")
plt.ylabel("ECG Amplitude")
plt.title("MIT-BIH Arrhythmia Database - ECG Record 100")

plt.grid(True)

plt.show()
# ============================================================
# ECG BEAT ANNOTATIONS
# ============================================================

plt.figure(figsize=(15, 5))

# ECG signal
plt.plot(
    time,
    signal,
    label="ECG Signal"
)

# Annotation positions
ann_samples = annotation.sample

# Only annotations within our 10-second segment
mask = ann_samples < samples

ann_x = ann_samples[mask] / record.fs
ann_y = signal[ann_samples[mask]]

# Mark beats
plt.scatter(
    ann_x,
    ann_y,
    marker="x",
    s=80,
    label="Beat Annotations"
)

plt.xlabel("Time (seconds)")
plt.ylabel("ECG Amplitude")
plt.title("ECG Signal with Beat Annotations")

plt.legend()
plt.grid(True)

plt.show()
# ============================================================
# RR INTERVAL AND HEART RATE ANALYSIS
# ============================================================

# Get annotated beat positions
beat_samples = annotation.sample

# Analyze first 30 seconds
duration = 30
limit = int(duration * record.fs)

beat_samples = beat_samples[beat_samples < limit]

# Difference between consecutive beats
rr_intervals_samples = np.diff(beat_samples)

# Convert samples to seconds
rr_intervals_seconds = rr_intervals_samples / record.fs

print("\nRR Interval Analysis")
print("-" * 40)

print("Number of detected beats:", len(beat_samples))
print("Number of RR intervals:", len(rr_intervals_seconds))

print("\nFirst 10 RR intervals (seconds):")
print(rr_intervals_seconds[:10])

# Average RR interval
mean_rr = np.mean(rr_intervals_seconds)

# Heart rate = 60 / RR interval
heart_rate = 60 / mean_rr

print("\nAverage RR Interval:",
      round(mean_rr, 3),
      "seconds")

print("Estimated Average Heart Rate:",
      round(heart_rate, 2),
      "BPM")
# ============================================================
# PART C - TEXTUAL MEDICAL DATA
# ============================================================

import re
from collections import Counter

clinical_note = """
Patient is a 55-year-old male presenting with chest pain,
shortness of breath and fatigue. The patient reports discomfort
during physical activity. Blood pressure is elevated and the
patient has a history of high cholesterol. ECG examination
shows abnormal cardiac activity. Further cardiovascular
evaluation is recommended.
"""

print("\nClinical Note:")
print(clinical_note)


# ============================================================
# TEXT CLEANING AND TOKENIZATION
# ============================================================

text = clinical_note.lower()

# Remove punctuation and numbers
clean_text = re.sub(r"[^a-zA-Z\s]", "", text)

# Tokenization
tokens = clean_text.split()

print("\nTokens:")
print(tokens)


# ============================================================
# WORD FREQUENCY
# ============================================================

word_frequency = Counter(tokens)

print("\nMost Common Words:")

for word, count in word_frequency.most_common(15):
    print(word, ":", count)


# ============================================================
# MEDICAL KEYWORD EXTRACTION
# ============================================================

medical_terms = [
    "chest",
    "pain",
    "shortness",
    "breath",
    "fatigue",
    "blood",
    "pressure",
    "cholesterol",
    "ecg",
    "cardiac",
    "cardiovascular"
]

found_terms = [
    term for term in medical_terms
    if term in tokens
]

print("\nMedical Terms Found:")
print(found_terms)
# ============================================================
# PART C - TEXTUAL MEDICAL DATA
# ============================================================

import re
from collections import Counter

clinical_note = """
Patient is a 55-year-old male presenting with chest pain,
shortness of breath and fatigue. The patient reports discomfort
during physical activity. Blood pressure is elevated and the
patient has a history of high cholesterol. ECG examination
shows abnormal cardiac activity. Further cardiovascular
evaluation is recommended.
"""

print("\n" + "=" * 60)
print("TEXTUAL MEDICAL DATA")
print("=" * 60)

print("\nClinical Note:")
print(clinical_note)

# Text cleaning
text = clinical_note.lower()
clean_text = re.sub(r"[^a-zA-Z\s]", "", text)

# Tokenization
tokens = clean_text.split()

print("\nNumber of Tokens:", len(tokens))
print("\nTokens:")
print(tokens)

# Word frequency
word_frequency = Counter(tokens)

print("\nMost Common Words:")
for word, count in word_frequency.most_common(15):
    print(f"{word}: {count}")

# Medical keyword extraction
medical_terms = [
    "chest",
    "pain",
    "shortness",
    "breath",
    "fatigue",
    "blood",
    "pressure",
    "cholesterol",
    "ecg",
    "cardiac",
    "cardiovascular"
]

found_terms = [
    term for term in medical_terms
    if term in tokens
]

print("\nMedical Terms Found:")
print(found_terms)


# ============================================================
# PART D - IMAGE DATA
# ============================================================

from PIL import Image

print("\n" + "=" * 60)
print("IMAGE DATA")
print("=" * 60)

# Create a synthetic medical-style grayscale image
image_array = np.zeros(
    (256, 256),
    dtype=np.uint8
)

y, x = np.ogrid[:256, :256]

# Main circular structure
center_x = 128
center_y = 128
radius = 80

circle = (
    (x - center_x) ** 2 +
    (y - center_y) ** 2
) <= radius ** 2

image_array[circle] = 180

# Additional structures
circle_left = (
    (x - 95) ** 2 +
    (y - 128) ** 2
) <= 25 ** 2

circle_right = (
    (x - 161) ** 2 +
    (y - 128) ** 2
) <= 25 ** 2

image_array[circle_left] = 100
image_array[circle_right] = 100

medical_image = Image.fromarray(image_array)

print("\nOriginal Image Size:", medical_image.size)
print("Image Mode:", medical_image.mode)

# Display original image
plt.figure(figsize=(6, 6))

plt.imshow(
    medical_image,
    cmap="gray"
)

plt.title("Synthetic Medical Image")
plt.axis("off")

plt.show()

# Resize image
resized_image = medical_image.resize((128, 128))

print("Resized Image Size:", resized_image.size)

# Display resized image
plt.figure(figsize=(5, 5))

plt.imshow(
    resized_image,
    cmap="gray"
)

plt.title("Resized Medical Image")
plt.axis("off")

plt.show()


# ============================================================
# PART E - MULTIMODAL MEDICAL DATA INTEGRATION
# ============================================================

print("\n" + "=" * 60)
print("MULTIMODAL MEDICAL DATA INTEGRATION")
print("=" * 60)

print("""
Patient
   |
   |--- Tabular Data
   |      Age
   |      Blood Pressure
   |      Cholesterol
   |      Clinical Features
   |
   |--- ECG Signal
   |      Heart Rate
   |      RR Intervals
   |      Beat Information
   |
   |--- Clinical Text
   |      Chest Pain
   |      Fatigue
   |      Symptoms
   |
   |--- Medical Image
          Imaging Features

             |
             v

      Feature Extraction
             |
             v

       Feature Fusion
             |
             v

        AI / ML Model
             |
             v

      Clinical Prediction
             |
             v

     Clinical Decision Support
""")


# ============================================================
# FINAL EXPERIMENT SUMMARY
# ============================================================

print("\n" + "=" * 60)
print("FINAL EXPERIMENT SUMMARY")
print("=" * 60)

print("""
TABULAR DATA
-------------
Dataset: Cleveland Heart Disease
Records: 303
Features: 13
Missing values handled: Yes
Model: Logistic Regression
Accuracy: 86.89%
ROC-AUC: 95.13%


SIGNAL DATA
-----------
Dataset: MIT-BIH Arrhythmia Database
Record: 100
Sampling Frequency: 360 Hz
Channels: 2
Detected Beats: 38
RR Intervals: 37
Average RR Interval: 0.794 seconds
Estimated Heart Rate: 75.59 BPM


TEXT DATA
---------
Clinical note processed
Tokenization performed
Word frequency calculated
Medical keywords extracted


IMAGE DATA
----------
Medical image loaded/created
Image displayed
Image resized


CONCLUSION
----------
The experiment successfully demonstrated the ingestion,
inspection and basic preprocessing of multiple healthcare
data modalities including tabular, signal, textual and
image data.

The experiment also demonstrated how these heterogeneous
modalities can be combined in a future multimodal AI
healthcare system for clinical prediction and decision
support.
""")