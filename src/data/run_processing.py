# src/data/processor.py
import pandas as pd
import numpy as np
from pathlib import Path
import logging
import argparse

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('data-processor')


def load_data(file_path):
    """Load data from a CSV file."""
    logger.info(f"Loading data from {file_path}")
    return pd.read_csv(file_path)


def clean_data(df):
    """Clean the dataset by handling missing values and outliers."""
    logger.info("Cleaning dataset")
    
    df_cleaned = df.copy()
    
    # Handle missing values
    for column in df_cleaned.columns:
        missing_count = df_cleaned[column].isnull().sum()
        if missing_count > 0:
            logger.info(f"Found {missing_count} missing values in {column}")
            
            if pd.api.types.is_numeric_dtype(df_cleaned[column]):
                median_value = df_cleaned[column].median()
                df_cleaned[column] = df_cleaned[column].fillna(median_value)
                logger.info(f"Filled missing values in {column} with median: {median_value}")
            else:
                mode_value = df_cleaned[column].mode()[0]
                df_cleaned[column] = df_cleaned[column].fillna(mode_value)
                logger.info(f"Filled missing values in {column} with mode: {mode_value}")
    
    # Handle outliers in price
    Q1 = df_cleaned['price'].quantile(0.25)
    Q3 = df_cleaned['price'].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    
    outliers = df_cleaned[(df_cleaned['price'] < lower_bound) | 
                          (df_cleaned['price'] > upper_bound)]
    
    if not outliers.empty:
        logger.info(f"Found {len(outliers)} outliers in price column")
        df_cleaned = df_cleaned[(df_cleaned['price'] >= lower_bound) & 
                                (df_cleaned['price'] <= upper_bound)]
        logger.info(f"Removed outliers. New dataset shape: {df_cleaned.shape}")
    
    return df_cleaned


def process_data(input_file: str, output_file: str):
    # Create output directory if it doesn't exist
    output_path = Path(output_file).parent
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Load data
    df = load_data(input_file)
    logger.info(f"Loaded data with shape: {df.shape}")
    
    # Clean data
    df_cleaned = clean_data(df)
    
    # Save processed data
    df_cleaned.to_csv(output_file, index=False)
    logger.info(f"Saved processed data to {output_file}")
    
    return df_cleaned


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process house data CSV")
    parser.add_argument(
        "--input", required=True, help="Path to input CSV file (relative or absolute)"
    )
    parser.add_argument(
        "--output", required=True, help="Path to save cleaned CSV file"
    )
    args = parser.parse_args()

    process_data(input_file=args.input, output_file=args.output)
