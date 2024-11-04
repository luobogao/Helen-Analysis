import os
import sys
import argparse
import glob
import pandas as pd

def parse_arguments():
    parser = argparse.ArgumentParser(description="Combine specific columns from multiple CSV files into one.")
    parser.add_argument("folder", type=str, help="Path to the folder containing CSV files.")
    parser.add_argument(
        "-o", "--output", type=str, default="combined.csv",
        help="Name of the output CSV file (default: combined.csv)"
    )
    return parser.parse_args()

def get_csv_files(folder):
    # Search for all CSV files in the folder (case-insensitive)
    csv_patterns = [os.path.join(folder, "*.csv"), os.path.join(folder, "*.CSV")]
    csv_files = []
    for pattern in csv_patterns:
        csv_files.extend(glob.glob(pattern))
    return csv_files

def validate_columns(df, required_columns, file_name):
    missing_columns = [col for col in required_columns if col not in df.columns]
    if missing_columns:
        print(f"Warning: The file '{file_name}' is missing columns: {missing_columns}. Skipping this file.")
        return False
    return True

def combine_csv_files(folder, output_file):
    csv_files = get_csv_files(folder)
    
    if not csv_files:
        print(f"No CSV files found in the folder '{folder}'. Exiting.")
        sys.exit(1)
    
    print(f"Found {len(csv_files)} CSV file(s) in '{folder}'.")
    
    # Define the columns to select
    required_columns = [
        "TimeStamp",
        "log",  # Added 'log' column
        "Delta_AF7", "Delta_AF8", "Delta_TP9", "Delta_TP10",
        "Theta_AF7", "Theta_AF8", "Theta_TP9", "Theta_TP10",
        "Alpha_AF7", "Alpha_AF8", "Alpha_TP9", "Alpha_TP10",
        "Beta_AF7", "Beta_AF8", "Beta_TP9", "Beta_TP10",
        "Gamma_AF7", "Gamma_AF8", "Gamma_TP9", "Gamma_TP10"
    ]
    
    combined_data = []
    for file in csv_files:
        try:
            df = pd.read_csv(file)
            if validate_columns(df, required_columns, os.path.basename(file)):
                selected_df = df[required_columns].copy()
                combined_data.append(selected_df)
                print(f"Processed file: {os.path.basename(file)}")
            else:
                continue
        except Exception as e:
            print(f"Error processing file '{file}': {e}. Skipping this file.")
    
    if not combined_data:
        print("No data to combine after processing all files. Exiting.")
        sys.exit(1)
    
    # Concatenate all dataframes
    combined_df = pd.concat(combined_data, ignore_index=True)
    
    # Save to CSV
    output_path = os.path.join(folder, output_file)
    try:
        combined_df.to_csv(output_path, index=False)
        print(f"\nSuccessfully combined {len(combined_data)} file(s) into '{output_path}'.")
    except Exception as e:
        print(f"Error saving combined CSV: {e}.")
        sys.exit(1)

def main():
    args = parse_arguments()
    folder = args.folder
    output_file = args.output
    
    if not os.path.isdir(folder):
        print(f"The folder '{folder}' does not exist or is not a directory. Exiting.")
        sys.exit(1)
    
    combine_csv_files(folder, output_file)

if __name__ == "__main__":
    main()
