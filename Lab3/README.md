# Lab 3 – Medical Text Data Preprocessing and Analysis

## Course Information

- **Course:** AI in Healthcare
- **Course Code:** CSET343
- **Semester:** VII
- **Year:** 4
- **Lab:** Lab 3
- **Dataset:** MTSamples Medical Transcriptions

---

## Objective

The objective of this lab is to perform data acquisition, cleaning, preprocessing, exploratory data analysis, feature engineering, and feature extraction on textual medical healthcare data.

The lab focuses on preparing medical transcription text for further healthcare Natural Language Processing (NLP) and machine learning tasks.

---

## Dataset

The **MTSamples Medical Transcriptions Dataset** contains medical transcription records belonging to different medical specialties.

The dataset is used to demonstrate preprocessing and analysis of real-world medical text data.

---

## Tasks Performed

### 1. Data Acquisition and Loading

- Loaded the MTSamples dataset from a CSV file.
- Used Pandas for reading and processing the dataset.

### 2. Dataset Inspection

The dataset was inspected using:

- First few records
- Dataset shape
- Column names
- Data types
- Missing values
- Duplicate records

### 3. Data Cleaning

The following cleaning operations were performed:

- Removed records with missing transcription text.
- Removed duplicate records.
- Cleaned unnecessary HTML content.
- Converted text to lowercase.
- Removed unwanted characters.
- Normalized extra spaces.

### 4. PHI De-identification

Medical text may contain Protected Health Information (PHI).

A basic rule-based de-identification process was applied to detect and replace:

- Email addresses
- Phone numbers
- URLs
- Dates
- Long numerical identifiers

These values were replaced with placeholders such as:

`[EMAIL]`, `[PHONE]`, `[URL]`, `[DATE]`, and `[ID]`.

---

## 5. Exploratory Data Analysis

Textual data was analyzed using:

- Text length distribution
- Word frequency
- Medical specialty/category distribution
- Most frequently occurring words

Visualizations were created using Matplotlib and Seaborn.

---

## 6. Feature Engineering

Two new features were created:

- **Word Count:** Number of words in each transcription.
- **Character Count:** Number of characters in each transcription.

These features help analyze the size and structure of medical documents.

---

## 7. Text Preprocessing

The transcription text was:

- Converted to lowercase
- Cleaned from HTML content
- Cleaned from unwanted characters
- Normalized for whitespace

This prepares the text for NLP-based processing.

---

## 8. TF-IDF Feature Extraction

TF-IDF (Term Frequency–Inverse Document Frequency) was used to convert medical text into numerical features.

TF-IDF gives higher importance to terms that are frequent in a document but less common across the entire collection of documents.

The extracted features can be used for further machine learning and NLP tasks.

---

## 9. Validation

The processed dataset was validated by checking:

- Final dataset shape
- Missing values
- Empty cleaned transcriptions
- Successfully generated TF-IDF features

The cleaned dataset was saved as:

`mtsamples_cleaned.csv`

---

## Tools and Technologies

- Python
- Pandas
- NumPy
- Matplotlib
- Seaborn
- Scikit-learn

---

## Conclusion

In this lab, medical transcription data was acquired, inspected, cleaned, de-identified, analyzed, and transformed into machine-learning-ready numerical features.

The preprocessing pipeline demonstrates how textual healthcare data can be prepared for Natural Language Processing and machine learning applications while considering healthcare-specific requirements such as privacy and data quality.