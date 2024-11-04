import pandas as pd
import sys
from sklearn.cluster import MiniBatchKMeans

def downsample_csv(input_path, output_path, target_points=10000):
    # Load data
    data = pd.read_csv(input_path)

    # Separate rows with 'log' values
    log_data = data.dropna(subset=['log'])
    non_log_data = data[data['log'].isna()]

    # Calculate target points after preserving log data
    remaining_points = target_points - len(log_data)
    clusters = non_log_data['cluster'].unique()
    points_per_cluster = remaining_points // len(clusters)

    final_data_list = [log_data[['log', 'X', 'Y', 'cluster']]]

    # Process each original 'cluster' separately
    for cluster_value in clusters:
        # Filter data for the current cluster
        cluster_data = non_log_data[non_log_data['cluster'] == cluster_value]

        # Apply clustering if more points than needed
        if len(cluster_data) > points_per_cluster:
            kmeans = MiniBatchKMeans(n_clusters=points_per_cluster, random_state=42, batch_size=1000)
            cluster_data['sub_cluster'] = kmeans.fit_predict(cluster_data[['X', 'Y']])
            reduced_cluster_data = cluster_data.groupby('sub_cluster').apply(lambda x: x.sample(1)).reset_index(drop=True)
        else:
            reduced_cluster_data = cluster_data

        final_data_list.append(reduced_cluster_data[['log', 'X', 'Y', 'cluster']])

    # Combine all clusters into final data
    final_data = pd.concat(final_data_list, ignore_index=True)
    final_data.to_csv(output_path, index=False)

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python downsample.py <input_csv_path>")
        sys.exit(1)
        
    input_path = sys.argv[1]
    output_path = input_path.replace(".csv", "_min.csv")
    downsample_csv(input_path, output_path)
    print(f"Downsampled CSV saved to: {output_path}")
