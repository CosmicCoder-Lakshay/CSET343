import os
import re
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from collections import Counter
from sklearn.feature_extraction.text import TfidfVectorizer


# ============================================================
# 1. LOAD DATASET
# ============================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

DATASET_PATH = os.path.join(
    BASE_DIR,
    "mtsamples.csv"
)

df = pd.read_csv(DATASET_PATH)

print("\n========== DATASET LOADED ==========")
print("Shape:", df.shape)


# ============================================================
# 2. INITIAL DATA INSPECTION
# ============================================================

print("\n========== FIRST 5 ROWS ==========")
print(df.head())

print("\n========== COLUMNS ==========")
print(df.columns.tolist())

print("\n========== DATA TYPES ==========")
print(df.dtypes)

print("\n========== MISSING VALUES ==========")
print(df.isnull().sum())

print("\n========== DUPLICATE ROWS ==========")
print(df.duplicated().sum())

print("\n========== DATASET INFORMATION ==========")
print(df.info())


# ============================================================
# 3. IDENTIFY TEXT COLUMN
# ============================================================

# MTSamples normally contains a transcription/text field.
# Automatically find a suitable text column.

possible_text_columns = [
    "transcription",
    "Transcription",
    "text",
    "Text"
]

text_column = None

for col in possible_text_columns:
    if col in df.columns:
        text_column = col
        break

if text_column is None:
    raise ValueError(
        "Text/transcription column not found. "
        "Please check the CSV column names."
    )

print("\nText column selected:", text_column)


# Remove rows where transcription is missing
df = df.dropna(subset=[text_column]).copy()


# ============================================================
# 4. REMOVE DUPLICATES
# ============================================================

before_duplicates = len(df)

df = df.drop_duplicates()

after_duplicates = len(df)

print("\n========== DUPLICATE REMOVAL ==========")
print("Rows before:", before_duplicates)
print("Rows after :", after_duplicates)
print("Duplicates removed:", before_duplicates - after_duplicates)


# ============================================================
# 5. PHI / DE-IDENTIFICATION
# ============================================================

def remove_phi(text):
    """
    Basic rule-based de-identification.
    Removes common direct identifiers such as:
    - email addresses
    - phone numbers
    - URLs
    - dates in common formats
    - some ID-like patterns
    """

    text = str(text)

    # Email addresses
    text = re.sub(
        r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b',
        '[EMAIL]',
        text
    )

    # Phone numbers
    text = re.sub(
        r'\b(?:\+?\d[\d\s().-]{7,}\d)\b',
        '[PHONE]',
        text
    )

    # URLs
    text = re.sub(
        r'https?://\S+|www\.\S+',
        '[URL]',
        text
    )

    # Dates such as 12/05/2020, 12-05-2020
    text = re.sub(
        r'\b\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\b',
        '[DATE]',
        text
    )

    # Long ID-like numbers
    text = re.sub(
        r'\b\d{6,}\b',
        '[ID]',
        text
    )

    return text


df["deidentified_text"] = df[text_column].apply(remove_phi)


# ============================================================
# 6. TEXT CLEANING
# ============================================================

def clean_text(text):

    text = str(text)

    # Convert to lowercase
    text = text.lower()

    # Remove HTML tags
    text = re.sub(r'<[^>]+>', ' ', text)

    # Keep alphabetic characters and spaces
    text = re.sub(r'[^a-z\s]', ' ', text)

    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text)

    return text.strip()


df["cleaned_text"] = df["deidentified_text"].apply(clean_text)


print("\n========== CLEANING EXAMPLE ==========")

print("\nOriginal:")
print(df[text_column].iloc[0][:500])

print("\nCleaned:")
print(df["cleaned_text"].iloc[0][:500])


# ============================================================
# 7. TEXT LENGTH ANALYSIS
# ============================================================

df["word_count"] = df["cleaned_text"].apply(
    lambda x: len(x.split())
)

df["character_count"] = df["cleaned_text"].apply(
    len
)

print("\n========== TEXT LENGTH STATISTICS ==========")

print(
    df[
        ["word_count", "character_count"]
    ].describe()
)


# ============================================================
# 8. EDA — TEXT LENGTH DISTRIBUTION
# ============================================================

plt.figure(figsize=(10, 5))

sns.histplot(
    df["word_count"],
    bins=30,
    kde=True
)

plt.title("Distribution of Medical Transcription Length")
plt.xlabel("Number of Words")
plt.ylabel("Number of Transcriptions")

plt.tight_layout()
plt.show()


# ============================================================
# 9. CATEGORY DISTRIBUTION
# ============================================================

# Find category column automatically

possible_category_columns = [
    "medical_specialty",
    "Medical Specialty",
    "category",
    "Category"
]

category_column = None

for col in possible_category_columns:
    if col in df.columns:
        category_column = col
        break

if category_column is not None:

    print("\n========== MEDICAL SPECIALTY DISTRIBUTION ==========")

    print(
        df[category_column]
        .value_counts()
        .head(15)
    )

    plt.figure(figsize=(12, 6))

    df[category_column].value_counts().head(15).plot(
        kind="bar"
    )

    plt.title("Top Medical Specialties")
    plt.xlabel("Medical Specialty")
    plt.ylabel("Number of Transcriptions")

    plt.xticks(rotation=45, ha="right")

    plt.tight_layout()
    plt.show()


# ============================================================
# 10. WORD FREQUENCY ANALYSIS
# ============================================================

all_words = []

for text in df["cleaned_text"]:

    words = text.split()

    all_words.extend(words)


word_counts = Counter(all_words)

most_common_words = word_counts.most_common(20)

print("\n========== TOP 20 WORDS ==========")

for word, count in most_common_words:
    print(f"{word}: {count}")


# ============================================================
# 11. WORD FREQUENCY VISUALIZATION
# ============================================================

words = [
    item[0]
    for item in most_common_words
]

counts = [
    item[1]
    for item in most_common_words
]

plt.figure(figsize=(12, 6))

sns.barplot(
    x=counts,
    y=words
)

plt.title("Top 20 Most Frequent Words")
plt.xlabel("Frequency")
plt.ylabel("Word")

plt.tight_layout()
plt.show()


# ============================================================
# 12. TF-IDF FEATURE EXTRACTION
# ============================================================

print("\n========== TF-IDF FEATURE EXTRACTION ==========")

tfidf = TfidfVectorizer(
    max_features=1000,
    min_df=2,
    max_df=0.95
)

tfidf_matrix = tfidf.fit_transform(
    df["cleaned_text"]
)

print("TF-IDF matrix shape:", tfidf_matrix.shape)

feature_names = tfidf.get_feature_names_out()

print("\nNumber of TF-IDF features:", len(feature_names))

print("\nSample TF-IDF features:")
print(feature_names[:20])


# ============================================================
# 13. FINAL MODEL-READY DATA
# ============================================================

print("\n========== FINAL DATA ==========")

print(
    df[
        [
            text_column,
            "cleaned_text",
            "word_count",
            "character_count"
        ]
    ].head()
)


# ============================================================
# 14. SAVE CLEANED DATA
# ============================================================

output_path = os.path.join(
    BASE_DIR,
    "mtsamples_cleaned.csv"
)

df.to_csv(
    output_path,
    index=False
)

print("\nCleaned dataset saved to:")
print(output_path)


# ============================================================
# 15. FINAL VALIDATION
# ============================================================

print("\n========== FINAL VALIDATION ==========")

print("Final shape:", df.shape)

print(
    "\nMissing values:"
)

print(
    df[
        [
            "cleaned_text",
            "word_count",
            "character_count"
        ]
    ].isnull().sum()
)

print(
    "\nEmpty cleaned texts:",
    (df["cleaned_text"].str.len() == 0).sum()
)

print("\n========================================")
print("LAB 3 - PART B COMPLETED SUCCESSFULLY!")
print("========================================")