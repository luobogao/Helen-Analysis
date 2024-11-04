import os
import sys
import argparse
import pandas as pd
from sklearn.decomposition import PCA
from openTSNE import TSNE
from sklearn.cluster import KMeans  # Import for clustering
import numpy as np
from datetime import datetime  # Import for timestamp

# ============================ CONFIGURABLE CONSTANTS ============================
# t-SNE Parameters
DEFAULT_PERPLEXITY = 140.0
DEFAULT_LEARNING_RATE = 200.0
DEFAULT_N_ITER = 1000
DEFAULT_PCA_COMPONENTS = 5
DEFAULT_SAMPLE_SIZE = None

# Data Filtering Constants
VALUE_THRESHOLD = 5.0  # Threshold for feature values

# Clustering Constants
DEFAULT_N_CLUSTERS = 20  # Number of clusters

# Script Behavior Constants
RANDOM_STATE = 42  # For reproducibility
VERBOSE = True     # Enable verbose output for t-SNE

# ============================ END OF CONFIGURABLE CONSTANTS =====================

def parse_arguments():
    parser = argparse.ArgumentParser(
        description=(
            "Perform clustering and t-SNE on a CSV file using all columns except TimeStamp and log, "
            "and save the results with TimeStamp, log, X, Y, and cluster."
        )
    )
    parser.add_argument(
        "input_csv",
        type=str,
        help="Path to the input CSV file containing TimeStamp, log, and feature columns."
    )
    parser.add_argument(
        "output_csv",
        type=str,
        help="Path to save the output CSV with TimeStamp, log, X, Y, and cluster."
    )
    parser.add_argument(
        "--perplexity",
        type=float,
        default=DEFAULT_PERPLEXITY,
        help=f"Perplexity parameter for t-SNE (default: {DEFAULT_PERPLEXITY})"
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=DEFAULT_LEARNING_RATE,
        help=f"Learning rate for t-SNE (default: {DEFAULT_LEARNING_RATE})"
    )
    parser.add_argument(
        "--n_iter",
        type=int,
        default=DEFAULT_N_ITER,
        help=f"Number of iterations for t-SNE optimization (default: {DEFAULT_N_ITER})"
    )
    parser.add_argument(
        "--pca_components",
        type=int,
        default=DEFAULT_PCA_COMPONENTS,
        help=f"Number of PCA components to reduce dimensionality before t-SNE (default: {DEFAULT_PCA_COMPONENTS})"
    )
    parser.add_argument(
        "--sample_size",
        type=int,
        default=DEFAULT_SAMPLE_SIZE,
        help="Number of samples to use for t-SNE and clustering. If not set, all data is used (default: None)"
    )
    parser.add_argument(
        "--timestamp_col",
        type=str,
        default="TimeStamp",
        help="Name of the TimeStamp column in the CSV (default: 'TimeStamp')"
    )
    parser.add_argument(
        "--log_col",
        type=str,
        default="log",
        help="Name of the log column in the CSV to include in the output (default: 'log')"
    )
    parser.add_argument(
        "--exclude_cols",
        type=str,
        nargs='*',
        default=[],
        help="List of additional column names to exclude from clustering and t-SNE (in addition to TimeStamp and log)"
    )
    parser.add_argument(
        "--n_clusters",
        type=int,
        default=DEFAULT_N_CLUSTERS,
        help=f"Number of clusters for KMeans (default: {DEFAULT_N_CLUSTERS})"
    )
    return parser.parse_args()

def load_data(csv_path, timestamp_col, log_col, exclude_cols):
    if not os.path.isfile(csv_path):
        print(f"Error: The file '{csv_path}' does not exist.")
        sys.exit(1)
    
    try:
        df = pd.read_csv(csv_path)
        print(f"Successfully loaded '{csv_path}'.")
    except Exception as e:
        print(f"Error reading '{csv_path}': {e}")
        sys.exit(1)
    
    # Check for TimeStamp column
    if timestamp_col not in df.columns:
        print(f"Error: The CSV file does not contain a '{timestamp_col}' column.")
        sys.exit(1)
    
    # Check for log column
    if log_col not in df.columns:
        print(f"Error: The CSV file does not contain a '{log_col}' column.")
        sys.exit(1)
    
    # Exclude specified columns along with TimeStamp and log
    feature_columns = [col for col in df.columns if col not in [timestamp_col, log_col] and col not in exclude_cols]
    if not feature_columns:
        print("Error: No feature columns found after excluding 'TimeStamp', 'log', and specified columns.")
        sys.exit(1)
    
    print(f"Selected {len(feature_columns)} feature columns for clustering and t-SNE.")
    return df[[timestamp_col, log_col] + feature_columns]

def preprocess_data(df, timestamp_col, log_col, pca_components=50, sample_size=None):
    # Handle missing values if any, excluding the 'log' column
    columns_to_drop_na = [col for col in df.columns if col != log_col]
    if df[columns_to_drop_na].isnull().any().any():
        print("Warning: Missing values detected. Rows with missing values (excluding 'log') will be dropped.")
        df = df.dropna(subset=columns_to_drop_na).reset_index(drop=True)  # Reset index after dropping NaNs
        print(f"Data shape after dropping missing values: {df.shape}")
    
    # Separate TimeStamp, log, and features
    timestamps = df[timestamp_col].reset_index(drop=True)
    logs = df[log_col].reset_index(drop=True)
    features = df.drop(columns=[timestamp_col, log_col]).copy()
    
    # Convert to numeric if not already
    non_numeric_cols = features.select_dtypes(exclude=[np.number]).columns.tolist()
    if non_numeric_cols:
        print(f"Found non-numeric columns: {non_numeric_cols}. Attempting to convert them to numeric.")
        for col in non_numeric_cols:
            features[col] = pd.to_numeric(features[col], errors='coerce')
        # After conversion, drop rows with NaN introduced by non-numeric data
        if features.isnull().values.any():
            print("Warning: Non-numeric data conversion resulted in NaN values. Dropping affected rows.")
            valid_rows = ~features.isnull().any(axis=1)
            num_dropped = len(features) - valid_rows.sum()
            features = features[valid_rows].reset_index(drop=True)
            timestamps = timestamps[valid_rows].reset_index(drop=True)
            logs = logs[valid_rows].reset_index(drop=True)
            print(f"Dropped {num_dropped} rows due to NaN values after conversion.")
    
    # =================== NEW: Filter out rows with out-of-range values ===================
    # Define the threshold
    threshold = VALUE_THRESHOLD
    print(f"Filtering out rows where any feature value is > {threshold} or < -{threshold}.")
    
    # Create a boolean mask where all feature values are within [-threshold, threshold]
    within_threshold = features.abs().le(threshold).all(axis=1)
    
    # Count how many rows will be kept or dropped
    num_total = features.shape[0]
    num_within = within_threshold.sum()
    num_dropped = num_total - num_within
    
    if num_dropped > 0:
        print(f"Dropping {num_dropped} rows due to feature values outside the range [-{threshold}, {threshold}].")
        features = features[within_threshold].reset_index(drop=True)
        timestamps = timestamps[within_threshold].reset_index(drop=True)
        logs = logs[within_threshold].reset_index(drop=True)
    else:
        print("No rows to drop based on feature value thresholds.")
    # =====================================================================================
    
    # Optionally sample the data
    if sample_size is not None and sample_size < features.shape[0]:
        print(f"Sampling {sample_size} out of {features.shape[0]} rows for clustering and t-SNE.")
        sampled_indices = np.random.choice(features.index, size=sample_size, replace=False)
        features = features.loc[sampled_indices].reset_index(drop=True)
        timestamps = timestamps.loc[sampled_indices].reset_index(drop=True)
        logs = logs.loc[sampled_indices].reset_index(drop=True)
    elif sample_size is not None:
        print(f"Requested sample_size {sample_size} exceeds data size {features.shape[0]}. Using all data.")
    
    # Apply PCA for dimensionality reduction
    if pca_components < features.shape[1]:
        print(f"Reducing dimensionality with PCA to {pca_components} components.")
        pca = PCA(n_components=pca_components, random_state=RANDOM_STATE)
        features_reduced = pca.fit_transform(features)
        print(f"PCA reduced data shape: {features_reduced.shape}")
    else:
        print("PCA not applied as pca_components >= number of features.")
        features_reduced = features.values
    
    return timestamps, logs, features_reduced

def perform_clustering(data, n_clusters=10):
    print(f"Starting KMeans clustering with {n_clusters} clusters.")
    kmeans = KMeans(n_clusters=n_clusters, random_state=RANDOM_STATE)
    try:
        clusters = kmeans.fit_predict(data)
        print("Clustering completed successfully.")
    except Exception as e:
        print(f"Error during clustering: {e}")
        sys.exit(1)
    return clusters

def perform_tsne(data, perplexity=30.0, learning_rate=200.0, n_iter=1000):
    print("Starting t-SNE dimensionality reduction with openTSNE...")
    tsne = TSNE(
        perplexity=perplexity,
        learning_rate=learning_rate,
        n_iter=n_iter,
        random_state=RANDOM_STATE,
        verbose=VERBOSE
    )
    try:
        tsne_results = tsne.fit(data)
        print("t-SNE completed successfully.")
    except Exception as e:
        print(f"Error during t-SNE computation: {e}")
        sys.exit(1)
    return tsne_results

def save_results(timestamps, logs, tsne_data, clusters, output_csv):
    if tsne_data.shape[0] != timestamps.shape[0] or tsne_data.shape[0] != logs.shape[0]:
        print("Error: The number of t-SNE results does not match the number of TimeStamp or log entries.")
        sys.exit(1)
    
    # Create a new DataFrame with TimeStamp, log, X, Y, and cluster
    tsne_df = pd.DataFrame({
        'TimeStamp': timestamps,
        'log': logs,
        'X': tsne_data[:, 0],
        'Y': tsne_data[:, 1],
        'cluster': clusters
    })
    
    # Add a processing timestamp
    processing_timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    tsne_df['processing_time'] = processing_timestamp
    
    try:
        tsne_df.to_csv(output_csv, index=False)
        print(f"Clustering and t-SNE results saved as '{output_csv}'.")
    except Exception as e:
        print(f"Error saving the results CSV: {e}")
        sys.exit(1)

def main():
    args = parse_arguments()
    input_csv = args.input_csv
    output_csv = args.output_csv
    perplexity = args.perplexity
    learning_rate = args.learning_rate
    n_iter = args.n_iter
    pca_components = args.pca_components
    sample_size = args.sample_size
    timestamp_col = args.timestamp_col
    log_col = args.log_col
    exclude_cols = args.exclude_cols
    n_clusters = args.n_clusters

    # Load and prepare data
    df = load_data(input_csv, timestamp_col, log_col, exclude_cols)

    # Preprocess data: handle missing values (excluding 'log'), convert to numeric, filter out out-of-range values, sample, apply PCA
    timestamps, logs, data_preprocessed = preprocess_data(df, timestamp_col, log_col, pca_components, sample_size)

    # Perform clustering on the preprocessed data
    clusters = perform_clustering(
        data_preprocessed,
        n_clusters=n_clusters
    )

    # Perform t-SNE on the preprocessed data
    tsne_results = perform_tsne(
        data_preprocessed,
        perplexity=perplexity,
        learning_rate=learning_rate,
        n_iter=n_iter
    )

    # Save the results with TimeStamp, log, X, Y, and cluster
    save_results(timestamps, logs, tsne_results, clusters, output_csv)

if __name__ == "__main__":
    main()
