import os
import sys
import argparse
import pandas as pd
import shutil

def parse_arguments():
    parser = argparse.ArgumentParser(
        description=(
            "Process a CSV file containing cluster assignments and logs, "
            "and extract non-null logs for each cluster into separate text files "
            "as well as a combined text file."
        )
    )
    parser.add_argument(
        "input_csv",
        type=str,
        help="Path to the input CSV file (ratios_xy file) containing 'cluster' and 'log' columns."
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="clusters_logs",
        help="Name of the output directory to save cluster log files (default: clusters_logs)."
    )
    return parser.parse_args()

def delete_output_directory(output_dir):
    """
    Deletes the output directory if it exists.
    """
    if os.path.exists(output_dir):
        try:
            shutil.rmtree(output_dir)
            print(f"Deleted existing directory '{output_dir}'.")
        except Exception as e:
            print(f"Error deleting directory '{output_dir}': {e}")
            sys.exit(1)
    else:
        print(f"No existing directory '{output_dir}' found. Proceeding to create a new one.")

def create_output_directory(output_dir):
    """
    Creates the output directory.
    """
    try:
        os.makedirs(output_dir)
        print(f"Created directory '{output_dir}'.")
    except Exception as e:
        print(f"Error creating directory '{output_dir}': {e}")
        sys.exit(1)

def load_data(csv_path):
    """
    Loads the CSV data and validates required columns.
    """
    if not os.path.isfile(csv_path):
        print(f"Error: The file '{csv_path}' does not exist.")
        sys.exit(1)
    
    try:
        df = pd.read_csv(csv_path)
        print(f"Successfully loaded '{csv_path}'.")
    except Exception as e:
        print(f"Error reading '{csv_path}': {e}")
        sys.exit(1)
    
    # Validate required columns
    required_columns = {'cluster', 'log'}
    if not required_columns.issubset(df.columns):
        missing = required_columns - set(df.columns)
        print(f"Error: The following required columns are missing in the CSV: {missing}")
        sys.exit(1)
    
    return df

def sanitize_cluster_label(cluster):
    """
    Sanitizes the cluster label to be used in filenames.
    Removes or replaces characters that are invalid in filenames.
    """
    # Convert to string and replace spaces with underscores
    cluster_str = str(cluster).replace(' ', '_')
    # Remove any characters that are not alphanumeric or underscores
    cluster_str = ''.join(c for c in cluster_str if c.isalnum() or c == '_')
    return cluster_str

def process_clusters(df, output_dir):
    """
    For each cluster, extracts non-null logs and saves them to a text file.
    Also compiles all clusters' logs into a combined.txt file.
    """
    unique_clusters = sorted(df['cluster'].unique())
    print(f"Found {len(unique_clusters)} unique clusters: {unique_clusters}")
    
    # Define the path for the combined text file
    combined_file = os.path.join(output_dir, "combined.txt")
    
    try:
        with open(combined_file, 'w', encoding='utf-8') as combined_f:
            for idx, cluster in enumerate(unique_clusters, start=1):
                # Filter rows for the current cluster
                cluster_df = df[df['cluster'] == cluster]
                print(f"Processing Cluster {cluster} ({idx}/{len(unique_clusters)}): {len(cluster_df)} entries found.")
                
                # Extract non-null 'log' entries
                non_null_logs = cluster_df['log'].dropna().astype(str).tolist()
                print(f"  - {len(non_null_logs)} non-null 'log' entries found.")
                
                if non_null_logs:
                    # Combine logs separated by commas
                    combined_logs = ','.join(non_null_logs)
                    
                    # Define the output file path for the individual cluster
                    sanitized_label = sanitize_cluster_label(cluster)
                    output_file = os.path.join(output_dir, f"cluster_{sanitized_label}.txt")
                    
                    # Write the combined logs to the individual cluster file
                    with open(output_file, 'w', encoding='utf-8') as f:
                        f.write(combined_logs)
                    print(f"  - Saved combined logs to '{output_file}'.")
                    
                    # Write to the combined.txt file with the cluster title
                    combined_f.write(f"CLUSTER {cluster}\n")
                    combined_f.write(combined_logs)
                    combined_f.write("\n\n\n")  # Three newlines to separate clusters
                else:
                    print(f"  - No non-null 'log' entries to save for Cluster {cluster}.")
        
        print(f"All clusters have been processed successfully. Combined logs saved to '{combined_file}'.")
    except Exception as e:
        print(f"Error during processing clusters: {e}")
        sys.exit(1)

def main():
    args = parse_arguments()
    input_csv = args.input_csv
    output_dir = args.output_dir
    
    # Delete existing output directory
    delete_output_directory(output_dir)
    
    # Create a new output directory
    create_output_directory(output_dir)
    
    # Load data
    df = load_data(input_csv)
    
    # Process each cluster and save logs
    process_clusters(df, output_dir)
    
    print("Script execution completed successfully.")

if __name__ == "__main__":
    main()
