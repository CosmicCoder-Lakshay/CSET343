import os
import numpy as np
import pandas as pd
import tensorflow as tf

from scikeras.wrappers import KerasClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer


# =========================
# 1. LOAD DATASET
# =========================

path = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "diabetes.csv"
)

df = pd.read_csv(path)

df.columns = df.columns.str.strip()

print("Shape:", df.shape)
print("Columns:", df.columns.tolist())
print("\nFirst 5 rows:\n", df.head())


# =========================
# 2. SEPARATE X AND Y
# =========================

X = df.drop("Outcome", axis=1)
y = df["Outcome"]


# =========================
# 3. HANDLE INVALID VALUES
# =========================

cols = [
    "Glucose",
    "BloodPressure",
    "SkinThickness",
    "Insulin",
    "BMI"
]

X[cols] = X[cols].replace(0, np.nan)


# =========================
# 4. IMPUTE MISSING VALUES
# =========================

imputer = SimpleImputer(strategy="median")

X = pd.DataFrame(
    imputer.fit_transform(X),
    columns=X.columns
)


# =========================
# 5. FEATURE SCALING
# =========================

scaler = StandardScaler()

X = scaler.fit_transform(X)


# =========================
# 6. TRAIN TEST SPLIT
# =========================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)


# =========================
# 7. CREATE ANN MODEL
# =========================

def create_model(
    learning_rate=0.001,
    init="glorot_uniform",
    activation="relu",
    dropout=0.2,
    neurons=16
):

    model = tf.keras.Sequential([
        
        tf.keras.layers.Input(
            shape=(X_train.shape[1],)
        ),

        tf.keras.layers.Dense(
            neurons,
            activation=activation,
            kernel_initializer=init
        ),

        tf.keras.layers.Dropout(dropout),

        tf.keras.layers.Dense(
            1,
            activation="sigmoid"
        )
    ])

    model.compile(
        optimizer=tf.keras.optimizers.Adam(
            learning_rate=learning_rate
        ),
        loss="binary_crossentropy",
        metrics=["accuracy"]
    )

    return model


# =========================
# 8. KERAS CLASSIFIER
# =========================

model = KerasClassifier(
    model=create_model,
    verbose=0
)


# =========================
# 9. GRID SEARCH
# =========================

parameters = [

    ("batch_size", [16, 32]),

    ("epochs", [10, 20]),

    ("model__learning_rate", [0.001, 0.01]),

    ("model__init",
     ["glorot_uniform", "he_uniform"]),

    ("model__activation",
     ["relu", "tanh"]),

    ("model__dropout",
     [0.2, 0.5]),

    ("model__neurons",
     [16, 32])
]


# =========================
# 10. TUNE PARAMETERS
# =========================

for name, values in parameters:

    print("\n==============================")
    print("Tuning:", name)
    print("==============================")

    grid = GridSearchCV(
        estimator=model,
        param_grid={name: values},
        cv=3,
        scoring="accuracy",
        n_jobs=1
    )

    grid.fit(X_train, y_train)

    print("Best Parameter:",
          grid.best_params_)

    print("Best CV Accuracy:",
          round(grid.best_score_, 4))


# =========================
# 11. FINAL MODEL
# =========================

model.set_params(
    batch_size=32,
    epochs=20,
    model__learning_rate=0.001,
    model__init="glorot_uniform",
    model__activation="relu",
    model__dropout=0.2,
    model__neurons=16
)

model.fit(X_train, y_train)


# =========================
# 12. TEST ACCURACY
# =========================

accuracy = model.score(
    X_test,
    y_test
)

print("\n==============================")
print("FINAL TEST ACCURACY:",
      round(accuracy, 4))
print("==============================")

print("\nLAB 7 COMPLETED SUCCESSFULLY!")