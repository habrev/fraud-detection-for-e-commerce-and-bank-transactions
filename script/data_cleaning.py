import pandas as pd

def clean_data(file_path, output_path):
    # Load data
    df = pd.read_csv(file_path)
    
    # Handle Missing Values
    df.dropna(inplace=True) 
    
    # Remove Duplicates
    df.drop_duplicates(inplace=True)
    
    # Correct Data Types
    df['signup_time'] = pd.to_datetime(df['signup_time'], errors='coerce')
    df['purchase_time'] = pd.to_datetime(df['purchase_time'], errors='coerce')
    df['purchase_value'] = pd.to_numeric(df['purchase_value'], errors='coerce')
    
    # Save cleaned data
    df.to_csv(output_path, index=False)
    
    print("Data cleaning completed successfully!")


