# Lab 6 – CNN for Chest X-Ray Classification

## Course
- Course Code: CSET343
- Course Name: AI in Healthcare
- Semester: VII
- Year: 4th Year

## Objective
To design and evaluate a CNN model for automated classification of chest X-ray images into Normal and Pneumonia classes.

## Dataset
Chest X-Ray Images Dataset.

Classes:
- NORMAL
- PNEUMONIA

## Tasks Performed

1. Loaded the chest X-ray dataset.
2. Displayed sample images from both classes.
3. Checked the number of images in each class.
4. Resized images to 128 × 128.
5. Converted images to grayscale.
6. Extracted image features using:
   - HOG (Histogram of Oriented Gradients)
   - LBP (Local Binary Patterns)
   - GLCM (Gray-Level Co-occurrence Matrix)
7. Prepared feature vectors.
8. Split the data into:
   - 70% Training
   - 15% Validation
   - 15% Testing
9. Built a CNN model using TensorFlow/Keras.
10. Evaluated the CNN using:
   - Accuracy
   - Precision
   - Recall
   - F1-score
   - AUROC
11. Generated a confusion matrix to analyze classification errors.

## Model
A Convolutional Neural Network was used for binary classification of chest X-ray images.

## Conclusion
The CNN model was trained to classify chest X-ray images as Normal or Pneumonia. The model performance was evaluated using classification metrics, AUROC, and a confusion matrix.