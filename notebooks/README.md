## Task 1 - Data Analysis and Preprocessing

### 1. Handle Missing Values
- Impute or drop missing values based on data characteristics.

### 2. Data Cleaning
- Remove duplicate records.
- Correct data types for consistency and accuracy.

### 3. Exploratory Data Analysis (EDA)
- **Univariate Analysis**: Examine individual features using statistical summaries and visualizations.
- **Bivariate Analysis**: Analyze relationships between variables to identify correlations and trends.

### 4. Merge Datasets for Geolocation Analysis
- Convert IP addresses to integer format.
- Merge `Fraud_Data.csv` with `IpAddress_to_Country.csv` for country-based fraud detection.

### 5. Feature Engineering
- **Transaction Frequency & Velocity**: Calculate the number of transactions per user within specific timeframes.
- **Time-Based Features**:
  - Extract `hour_of_day` from timestamps.
  - Extract `day_of_week` from timestamps.

### 6. Normalization and Scaling
- Apply Min-Max Scaling or Standardization to normalize numerical features.

### 7. Encode Categorical Features
- Convert categorical variables into numerical representations using techniques like One-Hot Encoding or Label Encoding.

# Task-2: Model Building and Training

## Overview
This project focuses on building and training machine learning models for fraud detection using two datasets:
- **Credit Card Dataset** (`Class` as the target variable)
- **Fraud Data Dataset** (`class` as the target variable)

The process includes data preparation, model training, evaluation, and MLOps steps for tracking and versioning.

---

## Data Preparation
1. **Feature and Target Separation:**
   - Extract feature variables (`X`) and target variable (`y`)
   - `Class` column for the credit card dataset
   - `class` column for the fraud data dataset

2. **Train-Test Split:**
   - Split the dataset into training (80%) and testing (20%) sets using `train_test_split` from `sklearn.model_selection`.

---

## Model Selection
Several machine learning models are used for performance comparison:
- **Logistic Regression**
- **Decision Tree**
- **Random Forest**
- **Gradient Boosting**
- **Multi-Layer Perceptron (MLP)**
- **Convolutional Neural Network (CNN)**
- **Recurrent Neural Network (RNN)**
- **Long Short-Term Memory (LSTM)**

---

## Model Training and Evaluation
1. **Train each model** on the training set.
2. **Evaluate model performance** using metrics such as:
   - Accuracy
   - Precision
   - Recall
   - F1-score
   - AUC-ROC
3. **Compare models** to select the best-performing one.

---

## MLOps Steps
### Experiment Tracking and Model Versioning
- **MLflow** is used to track:
  - Model parameters
  - Performance metrics
  - Experiment versions
- Run `mlflow.start_run()` to log experiments.
- Save models using `mlflow.sklearn.log_model()`.

