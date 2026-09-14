import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, label_binarize
from sklearn.pipeline import make_pipeline
from sklearn.impute import SimpleImputer

from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC

from sklearn.metrics import (
    accuracy_score, classification_report,
    confusion_matrix, roc_auc_score, roc_curve
)

# ================= DATASET =================

df = pd.read_csv(r"Lab 5\archive\dermatology_database_1.csv")

# Remove extra spaces from column names
df.columns = df.columns.str.strip()

print("Shape:", df.shape)
print("\nFirst 5 rows:\n", df.head())
print("\nStatistics:\n", df.describe())
print("\nMissing values:\n", df.isnull().sum())
print("\nDuplicates:", df.duplicated().sum())

df = df.drop_duplicates()

# Find target column automatically
target = None

for col in df.columns:
    if col.lower() in ["class", "target", "disease", "prognosis"]:
        target = col
        break

if target is None:
    target = df.columns[-1]

print("\nTarget Column:", target)
print("\nClass Distribution:\n", df[target].value_counts())

# ================= PREPROCESSING =================

X = df.drop(columns=[target])
y = df[target]

# Convert features to numeric
X = X.apply(pd.to_numeric, errors="coerce")

# Fill missing values
X = pd.DataFrame(
    SimpleImputer(strategy="median").fit_transform(X),
    columns=X.columns
)

# ================= VISUALIZATION =================

sns.countplot(x=y)
plt.title("Class Distribution")
plt.show()

X.iloc[:, :5].hist(figsize=(10, 6))
plt.suptitle("Feature Distributions")
plt.show()

plt.figure(figsize=(12, 8))
sns.heatmap(X.corr(), cmap="coolwarm")
plt.title("Correlation Heatmap")
plt.show()

# ================= TRAIN TEST SPLIT =================

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

# ================= MODELS =================

models = {
    "KNN": make_pipeline(
        StandardScaler(),
        KNeighborsClassifier(n_neighbors=5)
    ),

    "Random Forest": RandomForestClassifier(
        n_estimators=100,
        random_state=42
    ),

    "Logistic Regression": make_pipeline(
        StandardScaler(),
        LogisticRegression(max_iter=2000)
    ),

    "SVM": make_pipeline(
        StandardScaler(),
        SVC(probability=True, random_state=42)
    )
}

results = {}

# ================= TRAIN & EVALUATE =================

for name, model in models.items():

    model.fit(X_train, y_train)

    pred = model.predict(X_test)
    prob = model.predict_proba(X_test)

    accuracy = accuracy_score(y_test, pred)
    results[name] = accuracy

    print("\n==============================")
    print(name)
    print("==============================")

    print("Accuracy:", round(accuracy, 4))
    print("\nClassification Report:")
    print(classification_report(y_test, pred))

    # Confusion Matrix
    cm = confusion_matrix(y_test, pred)

    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
    plt.title(name + " Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.show()

    # Multiclass AUC
    classes = sorted(y.unique())
    y_test_bin = label_binarize(y_test, classes=classes)

    auc_score = roc_auc_score(
        y_test_bin,
        prob,
        multi_class="ovr",
        average="macro"
    )

    print("AUC Score:", round(auc_score, 4))

# ================= MODEL COMPARISON =================

print("\n========== MODEL COMPARISON ==========")

for name, score in results.items():
    print(name, ":", round(score, 4))

print("\nLAB 5 COMPLETED SUCCESSFULLY!")