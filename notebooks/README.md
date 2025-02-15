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

# Task 3 - Model Explainability

Model explainability is crucial for understanding, trust, and debugging in machine learning models. This project uses **SHAP** (Shapley Additive exPlanations) and **LIME** (Local Interpretable Model-agnostic Explanations) to interpret the fraud detection models.

### Using SHAP for Explainability
SHAP values provide a unified measure of feature importance, explaining the contribution of each feature to the prediction.


#### SHAP Plots
- **Summary Plot**: Provides an overview of the most important features.
- **Force Plot**: Visualizes the contribution of features for a single prediction.
- **Dependence Plot**: Shows the relationship between a feature and the model output.

### Using LIME for Explainability
LIME explains individual predictions by approximating the model locally with an interpretable model.

#### LIME Plots
- **Feature Importance Plot**: Shows the most influential features for a specific prediction.

## Task 4 - Model Deployment and API Development

### Setting Up the Flask API
1. **Create a new directory** for the project.
2. **Create a Python script** `serve_model.py` to serve the model using Flask.
3. **Create a `requirements.txt` file** to list dependencies.

### API Development
- Define API endpoints to serve fraud predictions.
- Test the API for accuracy and reliability.

### Dockerizing the Flask Application
Create a `Dockerfile` in the project directory:

### Integrate Logging
Use Flask-Logging to track:
- Incoming requests
- Errors
- Fraud predictions for continuous monitoring

## Task 5 - Build a Dashboard with Flask and Dash

### Overview
Create an interactive dashboard using **Dash** for visualizing fraud insights from the data. The **Flask backend** will serve data from the datasets, while **Dash** will be used to visualize insights.

### Flask API for Data Serving
- Create an endpoint that reads fraud data from a CSV file.
- Serve summary statistics and fraud trends through API endpoints.

### Dashboard Insights
The dashboard should display:
1. **Summary Statistics**:
   - Total transactions
   - Fraud cases
   - Fraud percentage
2. **Line Chart**:
   - Number of detected fraud cases over time
3. **Geographical Analysis**:
   - Locations where fraud is occurring
4. **Device and Browser Analysis**:
   - Bar chart comparing the number of fraud cases across different devices and browsers

This project ensures model transparency, reliable deployment, and insightful visualizations to detect and prevent fraudulent activities effectively.
