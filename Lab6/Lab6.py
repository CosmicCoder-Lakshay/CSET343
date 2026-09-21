import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from skimage.feature import hog, local_binary_pattern, graycomatrix, graycoprops

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, confusion_matrix
)

import tensorflow as tf
from tensorflow.keras import layers, models

# ================= SETTINGS =================

DATASET = r"Lab6\chest_xray\train"
SIZE = 128

images = []
labels = []
hog_features = []
lbp_features = []
glcm_features = []

# ================= LOAD DATA =================

classes = ["NORMAL", "PNEUMONIA"]

for label, cls in enumerate(classes):

    folder = os.path.join(DATASET, cls)

    for file in os.listdir(folder):

        path = os.path.join(folder, file)

        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)

        if img is None:
            continue

        img = cv2.resize(img, (SIZE, SIZE))

        images.append(img)
        labels.append(label)

        # HOG
        h = hog(
            img,
            orientations=9,
            pixels_per_cell=(16, 16),
            cells_per_block=(2, 2)
        )
        hog_features.append(h)

        # LBP
        lbp = local_binary_pattern(img, 8, 1, method="uniform")
        hist, _ = np.histogram(
            lbp.ravel(),
            bins=np.arange(11),
            range=(0, 10)
        )
        hist = hist.astype("float") / (hist.sum() + 1e-7)
        lbp_features.append(hist)

        # GLCM
        small = cv2.resize(img, (32, 32))
        glcm = graycomatrix(
            small,
            distances=[1],
            angles=[0],
            levels=256,
            symmetric=True,
            normed=True
        )

        g = [
            graycoprops(glcm, "contrast")[0, 0],
            graycoprops(glcm, "dissimilarity")[0, 0],
            graycoprops(glcm, "homogeneity")[0, 0],
            graycoprops(glcm, "energy")[0, 0],
            graycoprops(glcm, "correlation")[0, 0]
        ]

        glcm_features.append(g)


X = np.array(images) / 255.0
y = np.array(labels)

print("Total images:", len(X))
print("Normal:", np.sum(y == 0))
print("Pneumonia:", np.sum(y == 1))

# ================= SAMPLE IMAGES =================

plt.figure(figsize=(10, 5))

for i in range(8):
    plt.subplot(2, 4, i + 1)
    plt.imshow(X[i], cmap="gray")
    plt.title(classes[y[i]])
    plt.axis("off")

plt.tight_layout()
plt.show()

# ================= FEATURE VECTORS =================

features = np.hstack([
    np.array(hog_features),
    np.array(lbp_features),
    np.array(glcm_features)
])

print("Feature vector shape:", features.shape)

# ================= 70/15/15 SPLIT =================

X_train, X_temp, y_train, y_temp = train_test_split(
    X, y,
    test_size=0.30,
    random_state=42,
    stratify=y
)

X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp,
    test_size=0.50,
    random_state=42,
    stratify=y_temp
)

print("\nTrain:", len(X_train))
print("Validation:", len(X_val))
print("Test:", len(X_test))

# CNN input
X_train = X_train[..., np.newaxis]
X_val = X_val[..., np.newaxis]
X_test = X_test[..., np.newaxis]

# ================= CNN MODEL =================

model = models.Sequential([
    layers.Conv2D(32, (3, 3), activation="relu",
                  input_shape=(128, 128, 1)),
    layers.MaxPooling2D(),

    layers.Conv2D(64, (3, 3), activation="relu"),
    layers.MaxPooling2D(),

    layers.Conv2D(128, (3, 3), activation="relu"),
    layers.MaxPooling2D(),

    layers.Flatten(),

    layers.Dense(128, activation="relu"),
    layers.Dropout(0.5),

    layers.Dense(1, activation="sigmoid")
])

model.compile(
    optimizer="adam",
    loss="binary_crossentropy",
    metrics=["accuracy"]
)

# ================= TRAIN =================

history = model.fit(
    X_train,
    y_train,
    validation_data=(X_val, y_val),
    epochs=10,
    batch_size=32
)

# ================= EVALUATION =================

prob = model.predict(X_test).ravel()
pred = (prob >= 0.5).astype(int)

accuracy = accuracy_score(y_test, pred)
precision = precision_score(y_test, pred)
recall = recall_score(y_test, pred)
f1 = f1_score(y_test, pred)
auc = roc_auc_score(y_test, prob)

print("\n========== RESULTS ==========")
print("Accuracy :", accuracy)
print("Precision:", precision)
print("Recall   :", recall)
print("F1 Score :", f1)
print("AUROC    :", auc)

# ================= CONFUSION MATRIX =================

cm = confusion_matrix(y_test, pred)

sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=classes,
    yticklabels=classes
)

plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("CNN Confusion Matrix")
plt.show()

print("\nLAB 6 COMPLETED!")