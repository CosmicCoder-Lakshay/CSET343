import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Load the Heart Failure Clinical Records Dataset
file_path = "archive/heart_failure_clinical_records_dataset.csv"

df = pd.read_csv(file_path)

# Display first 5 rows
print("First 5 rows of the dataset:")
print(df.head())

# Display dataset information
print("\nDataset Shape:")
print(df.shape)

print("\nDataset Information:")
df.info()

print("\nData Types:")
print(df.dtypes)

print("\nMissing Values:")
print(df.isnull().sum())

# Visualize missing values using a heatmap
plt.figure(figsize=(12, 6))
sns.heatmap(df.isnull(), cbar=False, cmap="viridis")
plt.title("Missing Values Heatmap")
plt.show()

# Select numerical columns
numerical_columns = df.select_dtypes(include=['int64', 'float64']).columns

# Create histograms for numerical features
df[numerical_columns].hist(figsize=(15, 12), bins=20)
plt.suptitle("Histograms of Numerical Features")
plt.tight_layout()
plt.show()

# Create boxplots for numerical features
plt.figure(figsize=(15, 8))
sns.boxplot(data=df[numerical_columns])
plt.title("Boxplots of Numerical Features")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()
# Summary Statistics
print("\nSummary Statistics:")
print(df.describe())

# Check target variable distribution
print("\nDEATH_EVENT Distribution:")
print(df["DEATH_EVENT"].value_counts())

# Visualize DEATH_EVENT distribution
plt.figure(figsize=(6, 5))

sns.countplot(data=df, x="DEATH_EVENT")

plt.title("Distribution of DEATH_EVENT")
plt.xlabel("DEATH_EVENT")
plt.ylabel("Number of Patients")

plt.show()

# Relationship between Age and DEATH_EVENT
plt.figure(figsize=(8, 5))

sns.boxplot(data=df, x="DEATH_EVENT", y="age")

plt.title("Age Distribution by DEATH_EVENT")
plt.xlabel("DEATH_EVENT (0 = Survived, 1 = Death)")
plt.ylabel("Age")

plt.show()

# Correlation Analysis
import seaborn as sns
import matplotlib.pyplot as plt

plt.figure(figsize=(14, 10))
sns.heatmap(df.corr(), annot=True, cmap="coolwarm", fmt=".2f")
plt.title("Correlation Heatmap")
plt.tight_layout()
plt.show()


# Top Correlations with DEATH_EVENT
print("\nCorrelation with DEATH_EVENT:")
print(df.corr()["DEATH_EVENT"].sort_values(ascending=False))


# Scatter Plot: Age vs Serum Creatinine
plt.figure(figsize=(8, 5))
sns.scatterplot(
    data=df,
    x="age",
    y="serum_creatinine",
    hue="DEATH_EVENT"
)
plt.title("Age vs Serum Creatinine by DEATH_EVENT")
plt.tight_layout()
plt.show()


# Scatter Plot: Ejection Fraction vs Serum Creatinine
plt.figure(figsize=(8, 5))
sns.scatterplot(
    data=df,
    x="ejection_fraction",
    y="serum_creatinine",
    hue="DEATH_EVENT"
)
plt.title("Ejection Fraction vs Serum Creatinine")
plt.tight_layout()
plt.show()


# Average Values Grouped by DEATH_EVENT
print("\nMean values grouped by DEATH_EVENT:")
print(df.groupby("DEATH_EVENT").mean())


# Final Dataset Summary
print("\nFinal Dataset Summary:")
print("Total Patients:", df.shape[0])
print("Total Features:", df.shape[1])
print("Missing Values:", df.isnull().sum().sum())

print("\nEDA Completed Successfully!")
# ==========================================
# DATA CLEANING AND FEATURE ENGINEERING
# ==========================================

import numpy as np
from sklearn.preprocessing import MinMaxScaler

# Create a copy of the original dataset
cleaned_df = df.copy()

print("\n" + "=" * 50)
print("DATA CLEANING AND FEATURE ENGINEERING")
print("=" * 50)


# ==========================================
# 1. IMPUTE MISSING VALUES
# ==========================================

print("\nMissing Values Before Cleaning:")
print(cleaned_df.isnull().sum())

# Select numerical columns
numerical_columns = cleaned_df.select_dtypes(
    include=['int64', 'float64']
).columns

# Fill missing numerical values with median
for column in numerical_columns:
    cleaned_df[column] = cleaned_df[column].fillna(
        cleaned_df[column].median()
    )

# Select categorical columns
categorical_columns = cleaned_df.select_dtypes(
    include=['object']
).columns

# Fill missing categorical values with mode
for column in categorical_columns:
    cleaned_df[column] = cleaned_df[column].fillna(
        cleaned_df[column].mode()[0]
    )

print("\nMissing Values After Imputation:")
print(cleaned_df.isnull().sum())


# ==========================================
# 2. REMOVE OUTLIERS USING IQR METHOD
# ==========================================

print("\nDataset Shape Before Outlier Removal:")
print(cleaned_df.shape)

# Columns where outliers will be removed
outlier_columns = [
    'age',
    'ejection_fraction',
    'serum_creatinine'
]

for column in outlier_columns:

    Q1 = cleaned_df[column].quantile(0.25)
    Q3 = cleaned_df[column].quantile(0.75)

    IQR = Q3 - Q1

    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR

    cleaned_df = cleaned_df[
        (cleaned_df[column] >= lower_bound) &
        (cleaned_df[column] <= upper_bound)
    ]

print("\nDataset Shape After Outlier Removal:")
print(cleaned_df.shape)


# ==========================================
# 3. CHECK AND FIX INVALID VALUES
# ==========================================

print("\nChecking for Invalid Values...")

# Remove invalid ages
cleaned_df = cleaned_df[cleaned_df['age'] > 0]

# Remove invalid ejection fraction values
cleaned_df = cleaned_df[
    (cleaned_df['ejection_fraction'] >= 0) &
    (cleaned_df['ejection_fraction'] <= 100)
]

# Remove invalid serum creatinine values
cleaned_df = cleaned_df[
    cleaned_df['serum_creatinine'] >= 0
]

print("Invalid values checked and corrected.")


# ==========================================
# 4. FEATURE ENGINEERING - AGE GROUP
# ==========================================

cleaned_df['age_group'] = pd.cut(
    cleaned_df['age'],
    bins=[0, 40, 60, np.inf],
    labels=['<40', '40-60', '>60']
)

print("\nAge Group Feature Created:")
print(cleaned_df['age_group'].value_counts())


# ==========================================
# 5. FEATURE ENGINEERING - RISK SCORE
# ==========================================

# Normalize required columns temporarily
ejection_norm = (
    cleaned_df['ejection_fraction'] -
    cleaned_df['ejection_fraction'].min()
) / (
    cleaned_df['ejection_fraction'].max() -
    cleaned_df['ejection_fraction'].min()
)

creatinine_norm = (
    cleaned_df['serum_creatinine'] -
    cleaned_df['serum_creatinine'].min()
) / (
    cleaned_df['serum_creatinine'].max() -
    cleaned_df['serum_creatinine'].min()
)

# Create risk score
cleaned_df['risk_score'] = (
    ejection_norm * creatinine_norm
)

print("\nRisk Score Created:")
print(cleaned_df['risk_score'].head())


# ==========================================
# 6. ENCODE CATEGORICAL VARIABLES
# ==========================================

print("\nEncoding Categorical Variables...")

# Convert age_group into numerical labels
age_group_mapping = {
    '<40': 0,
    '40-60': 1,
    '>60': 2
}

cleaned_df['age_group'] = cleaned_df['age_group'].map(
    age_group_mapping
)

# sex and smoking are already binary,
# but converting them explicitly to integer
cleaned_df['sex'] = cleaned_df['sex'].astype(int)
cleaned_df['smoking'] = cleaned_df['smoking'].astype(int)

print("Categorical Variables Encoded Successfully.")


# ==========================================
# 7. NORMALIZE FEATURES USING MINMAXSCALER
# ==========================================

print("\nNormalizing Numerical Features...")

columns_to_scale = [
    'age',
    'anaemia',
    'creatinine_phosphokinase',
    'diabetes',
    'ejection_fraction',
    'high_blood_pressure',
    'platelets',
    'serum_creatinine',
    'serum_sodium',
    'sex',
    'smoking',
    'time',
    'risk_score'
]

scaler = MinMaxScaler()

cleaned_df[columns_to_scale] = scaler.fit_transform(
    cleaned_df[columns_to_scale]
)

print("Normalization Completed.")


# ==========================================
# 8. SUMMARY STATISTICS BEFORE CLEANING
# ==========================================

print("\n" + "=" * 50)
print("SUMMARY STATISTICS BEFORE CLEANING")
print("=" * 50)

print(df.describe())


# ==========================================
# 9. SUMMARY STATISTICS AFTER CLEANING
# ==========================================

print("\n" + "=" * 50)
print("SUMMARY STATISTICS AFTER CLEANING")
print("=" * 50)

print(cleaned_df.describe())


# ==========================================
# 10. VISUALIZE CLEANED DATA
# ==========================================

import matplotlib.pyplot as plt
import seaborn as sns

# Histograms after cleaning
cleaned_df[
    ['age', 'ejection_fraction', 'serum_creatinine']
].hist(figsize=(12, 5), bins=20)

plt.suptitle("Histograms After Data Cleaning")
plt.tight_layout()
plt.show()


# Boxplots after cleaning
plt.figure(figsize=(10, 6))

sns.boxplot(
    data=cleaned_df[
        ['age', 'ejection_fraction', 'serum_creatinine']
    ]
)

plt.title("Boxplots After Data Cleaning")
plt.show()


# ==========================================
# 11. FINAL MISSING VALUES CHECK
# ==========================================

print("\n" + "=" * 50)
print("FINAL MISSING VALUES CHECK")
print("=" * 50)

print(cleaned_df.isnull().sum())

total_missing = cleaned_df.isnull().sum().sum()

print("\nTotal Missing Values:", total_missing)

if total_missing == 0:
    print("\nSUCCESS: No missing values remain in the cleaned dataset.")
else:
    print("\nSome missing values are still present.")


# ==========================================
# 12. FINAL DATASET INFORMATION
# ==========================================

print("\n" + "=" * 50)
print("FINAL CLEANED DATASET")
print("=" * 50)

print("\nFinal Dataset Shape:")
print(cleaned_df.shape)

print("\nFirst 5 Rows of Cleaned Dataset:")
print(cleaned_df.head())

print("\nData Cleaning and Feature Engineering Completed Successfully!")