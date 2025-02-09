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