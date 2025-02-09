import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def perform_eda(df):
    # Univariate Analysis
    def univariate_analysis(df):
        print("Univariate Analysis:")
        
        # Summary statistics for numerical columns
        print("\nSummary Statistics:")
        print(df.describe())
        
        # Plotting histograms for numerical columns
        num_columns = df.select_dtypes(include=['float64', 'int64']).columns
        for col in num_columns:
            plt.figure(figsize=(8, 6))
            sns.histplot(df[col], kde=True, bins=30, color='blue')
            plt.title(f"Distribution of {col}")
            plt.show()
        
        # Plotting countplot for categorical columns
        cat_columns = df.select_dtypes(include=['object', 'category']).columns
        for col in cat_columns:
            plt.figure(figsize=(8, 6))
            sns.countplot(x=df[col], palette='Set2')
            plt.title(f"Countplot of {col}")
            plt.show()
        
    # Bivariate Analysis
    def bivariate_analysis(df):
        print("\nBivariate Analysis:")
        
        # Correlation heatmap for numerical columns
        num_columns = df.select_dtypes(include=['float64', 'int64']).columns
        corr = df[num_columns].corr()
        plt.figure(figsize=(10, 8))
        sns.heatmap(corr, annot=True, cmap='coolwarm', fmt='.2f', linewidths=0.5)
        plt.title("Correlation Heatmap")
        plt.show()

        # Scatter plots for numerical vs numerical columns
        for col1 in num_columns:
            for col2 in num_columns:
                if col1 != col2:
                    plt.figure(figsize=(8, 6))
                    sns.scatterplot(x=df[col1], y=df[col2], color='green')
                    plt.title(f"Scatter Plot: {col1} vs {col2}")
                    plt.show()
        
        # Boxplot for numerical vs categorical columns
        for col in cat_columns:
            for num_col in num_columns:
                plt.figure(figsize=(8, 6))
                sns.boxplot(x=df[col], y=df[num_col], palette='Set2')
                plt.title(f"Boxplot: {num_col} vs {col}")
                plt.show()

    # Run both univariate and bivariate analysis
    univariate_analysis(df)
    bivariate_analysis(df)

