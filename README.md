# A Clojure-Based Machine Learning Library

This project for MAT340 focuses on building machine learning models from scratch using Clojure, with only `clojure.core.matrix` and `clojure.data.csv` as external libraries. It includes the following capabilities:

- Data Preprocessing
  - Import dataset from CSV ✓
  - Splitting dataset
  - Scalers ✓
- Features Extraction
  - PCA (Principal Component Analysis) ✓
  - LLE (Locally Linear Embedding) `?`
- Models
  - Perceptron ✓
  - Adaline ✓
  - Logistic Regression ✓
  - KNN (K-nearest Neighbor) ✓
  - Decision Tree `?`
  - SVM (Support Vector Machine) `https://stackoverflow.com/questions/62242579/implementing-svm-rbf ?`
- Metrics
  - Accuracy ✓
  - Confusion Matrix ✓
  - Precision ✓
  - Recall ✓
  - F1-score ✓
