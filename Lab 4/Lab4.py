import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, classification_report,
    confusion_matrix, roc_curve, roc_auc_score
)

# Load dataset
cols = ["id","diagnosis",
"radius_mean","texture_mean","perimeter_mean","area_mean","smoothness_mean",
"compactness_mean","concavity_mean","concave_points_mean","symmetry_mean",
"fractal_dimension_mean","radius_se","texture_se","perimeter_se","area_se",
"smoothness_se","compactness_se","concavity_se","concave_points_se",
"symmetry_se","fractal_dimension_se","radius_worst","texture_worst",
"perimeter_worst","area_worst","smoothness_worst","compactness_worst",
"concavity_worst","concave_points_worst","symmetry_worst",
"fractal_dimension_worst"]

df = pd.read_csv(r"Lab 4\Dataset\wdbc.data", header=None, names=cols)

print(df.head())
print("\nShape:", df.shape)
print("\nStatistics:\n", df.describe())
print("\nMissing Values:\n", df.isnull().sum())
print("\nDuplicates:", df.duplicated().sum())
print("\nClass Distribution:\n", df.diagnosis.value_counts())

# Visualization
sns.countplot(x="diagnosis", data=df)
plt.title("Class Distribution")
plt.show()

sns.heatmap(df.iloc[:, 2:].corr(), cmap="coolwarm")
plt.title("Correlation Heatmap")
plt.show()

df[["radius_mean","texture_mean","area_mean"]].hist()
plt.show()

# Preprocessing
df["Outcome"] = df["diagnosis"].map({"B": 0, "M": 1})

X = df.iloc[:, 2:-1]
y = df["Outcome"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Logistic Regression
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
y_prob = model.predict_proba(X_test)[:, 1]

# Evaluation
print("\nAccuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n",
      classification_report(y_test, y_pred))

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt="d",
            xticklabels=["Benign","Malignant"],
            yticklabels=["Benign","Malignant"])
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()

# ROC Curve
fpr, tpr, _ = roc_curve(y_test, y_prob)
auc = roc_auc_score(y_test, y_prob)

plt.plot(fpr, tpr, label=f"AUC = {auc:.3f}")
plt.plot([0,1], [0,1], "--")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve")
plt.legend()
plt.show()

print("AUC Score:", auc)