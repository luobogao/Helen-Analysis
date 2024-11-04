import os
import sys
import argparse
import pandas as pd
import numpy as np

def parse_arguments():
    parser = argparse.ArgumentParser(
        description=(
            "Transform vectors by calculating ratios across different locations for each frequency band, "
            "apply logarithmic transformation to the ratios, and save the results to a new CSV file."
        )
    )
    parser.add_argument(
        "input_csv",
        type=str,
        help="Path to the cleaned input CSV file containing TimeStamp, log, and frequency band columns."
    )
    parser.add_argument(
        "output_csv",
        type=str,
        help="Path to save the transformed output CSV file with TimeStamp, log, and new log-transformed vector ratio columns."
    )
    return parser.parse_args()

def validate_columns(df, required_columns, file_name):
    missing_columns = [col for col in required_columns if col not in df.columns]
    if missing_columns:
        print(f"Error: The file '{file_name}' is missing the following required columns: {missing_columns}")
        sys.exit(1)

def transform_vectors(df):
    bands = ['Delta', 'Theta', 'Alpha', 'Beta', 'Gamma']
    locations = ['AF7', 'AF8', 'TP9', 'TP10']
    new_columns = {}

    for band in bands:
        # Define column names for the current band
        band_columns = {
            'AF7': f"{band}_AF7",
            'AF8': f"{band}_AF8",
            'TP9': f"{band}_TP9",
            'TP10': f"{band}_TP10"
        }

        # Check if all required columns for the current band exist
        if not all(col in df.columns for col in band_columns.values()):
            print(f"Warning: Not all required columns for the '{band}' band are present. Skipping transformations for this band.")
            continue

        # Calculate new vector ratios
        try:
            # Ratio: TP10 / TP9
            ratio_tp10_tp9 = df[band_columns['TP10']] / df[band_columns['TP9']]
            # Apply log transformation, handle non-positive values
            log_tp10_tp9 = np.log(ratio_tp10_tp9.replace({0: np.nan, -np.inf: np.nan}))
            new_columns[f"{band}_TP10_TP9_log"] = log_tp10_tp9

            # Ratio: AF8 / AF7
            ratio_af8_af7 = df[band_columns['AF8']] / df[band_columns['AF7']]
            # Apply log transformation, handle non-positive values
            log_af8_af7 = np.log(ratio_af8_af7.replace({0: np.nan, -np.inf: np.nan}))
            new_columns[f"{band}_AF8_AF7_log"] = log_af8_af7

            # Ratio: AF8 / TP10
            ratio_af8_tp10 = df[band_columns['AF8']] / df[band_columns['TP10']]
            # Apply log transformation, handle non-positive values
            log_af8_tp10 = np.log(ratio_af8_tp10.replace({0: np.nan, -np.inf: np.nan}))
            new_columns[f"{band}_AF8_TP10_log"] = log_af8_tp10

            # Ratio: AF7 / TP9
            ratio_af7_tp9 = df[band_columns['AF7']] / df[band_columns['TP9']]
            # Apply log transformation, handle non-positive values
            log_af7_tp9 = np.log(ratio_af7_tp9.replace({0: np.nan, -np.inf: np.nan}))
            new_columns[f"{band}_AF7_TP9_log"] = log_af7_tp9

        except Exception as e:
            print(f"Error while transforming vectors for band '{band}': {e}")
            continue

    # Create a new DataFrame with TimeStamp, log, and new columns
    transformed_df = pd.DataFrame()
    transformed_df['TimeStamp'] = df['TimeStamp']
    
    # **Include the 'log' column**
    transformed_df['log'] = df['log']

    for col_name, col_data in new_columns.items():
        transformed_df[col_name] = col_data

    # Inform the user about any NaN values introduced by log transformation
    total_nans = transformed_df.isna().sum().sum()
    if total_nans > 0:
        print(f"Warning: {total_nans} NaN values detected in the transformed ratios due to non-positive original ratios.")

    return transformed_df

def main():
    args = parse_arguments()
    input_csv = args.input_csv
    output_csv = args.output_csv

    # Check if input file exists
    if not os.path.isfile(input_csv):
        print(f"Error: The input file '{input_csv}' does not exist.")
        sys.exit(1)

    # Read the input CSV
    try:
        df = pd.read_csv(input_csv)
        print(f"Successfully loaded '{input_csv}'.")
    except Exception as e:
        print(f"Error reading '{input_csv}': {e}")
        sys.exit(1)

    # Define required columns
    required_columns = ['TimeStamp', 'log']  # **Added 'log' as a required column**
    bands = ['Delta', 'Theta', 'Alpha', 'Beta', 'Gamma']
    locations = ['AF7', 'AF8', 'TP9', 'TP10']
    for band in bands:
        for loc in locations:
            required_columns.append(f"{band}_{loc}")

    # Validate required columns
    validate_columns(df, required_columns, input_csv)

    # Transform vectors and apply log transformation
    transformed_df = transform_vectors(df)

    # Save the transformed DataFrame to a new CSV
    try:
        transformed_df.to_csv(output_csv, index=False)
        print(f"Transformed data successfully saved to '{output_csv}'.")
    except Exception as e:
        print(f"Error saving transformed data to '{output_csv}': {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
