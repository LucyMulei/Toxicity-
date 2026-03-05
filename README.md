# Toxicity Classification using Machine Learning

## Project Overview

This project builds a machine learning model that predicts whether a sample is **Toxic** or **Non-Toxic** based on numerical features.

The goal is to demonstrate a complete machine learning workflow including:

* Data preprocessing
* Feature selection
* Model training
* Cross-validation
* Model evaluation

The final model uses a **Random Forest classifier** and focuses on improving performance through dimensionality reduction and feature importance analysis.

---

## Dataset

The dataset contains **171 observations** with a large number of features extracted from samples.

Target variable:

* **Class**

  * `0` → NonToxic
  * `1` → Toxic

Class distribution:

| Class    | Proportion |
| -------- | ---------- |
| NonToxic | 0.67       |
| Toxic    | 0.33       |

This indicates a **moderate class imbalance**, which is an important factor when evaluating model performance.

---

## Project Workflow

### 1. Data Loading

The dataset was loaded into a pandas DataFrame and inspected to understand the structure and class distribution.

---

### 2. Feature Selection

Because the dataset contained **more than 1000 features**, dimensionality reduction was necessary to prevent overfitting.

The following steps were applied:

#### a) Low Variance Filtering

Features with extremely low variance were removed since they provide little information for classification.

#### b) Correlation Filtering

Highly correlated features were removed to reduce redundancy.

Result:

Original feature space → **1189 features**

After filtering → **472 features**

---

### 3. Feature Importance Selection

A **Random Forest model** was used to compute feature importance scores.

Features with very small importance values (<0.001) were removed.

Final dataset size:

**171 samples × 50 features**

This step reduces noise and keeps only the most predictive variables.

---

### 4. Model Training

The base model used is:

**Random Forest Classifier**

Reasons for choosing Random Forest:

* Handles high-dimensional data well
* Resistant to overfitting
* Provides feature importance
* Works well with nonlinear relationships

Cross-validation was used to evaluate model performance.

---

### 5. Model Evaluation

Performance metrics:

* Precision
* Recall
* F1 Score
* ROC-AUC

Final evaluation on the test set:

| Class    | Precision | Recall | F1   |
| -------- | --------- | ------ | ---- |
| NonToxic | 0.72      | 0.75   | 0.73 |
| Toxic    | 0.40      | 0.36   | 0.38 |

Overall Accuracy: **0.63**

ROC-AUC: **0.659**

---

## Interpretation of Results

The model performs well at identifying **Non-Toxic samples**, but struggles more with identifying **Toxic samples**.

This happens because:

* The dataset is relatively small (171 samples)
* There is class imbalance
* The Toxic class has fewer examples

Despite this, the model still captures meaningful patterns and achieves a **moderate ROC-AUC score (~0.66)**.

Future improvements could include:

* More training data
* Advanced ensemble models (XGBoost, LightGBM)
* Handling class imbalance using SMOTE or class weighting
* Hyperparameter tuning

---

## Technologies Used

* Python
* Pandas
* NumPy
* Scikit-learn
* Jupyter Notebook

---

## How to Run the Project

1. Clone the repository

```
git clone https://github.com/yourusername/your-repository-name.git
```

2. Navigate into the project directory

```
cd your-repository-name
```

3. Install required libraries

```
pip install pandas numpy scikit-learn matplotlib seaborn
```

4. Open the notebook

```
jupyter notebook
```

Run the notebook to reproduce the results.

---

## Key Learning Outcomes

This project demonstrates:

* Feature reduction in high-dimensional datasets
* Using ensemble models for classification
* Evaluating models using cross-validation
* Interpreting feature importance
* Communicating machine learning results clearly

---


Data Science and Analytics Student
