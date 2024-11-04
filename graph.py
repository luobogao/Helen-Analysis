import os
import sys
import argparse
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

# ============================ CONFIGURABLE CONSTANTS ============================
# Plot Appearance Constants
MARKER_SIZE = 100              # Size of the scatter plot markers
MARKER_OPACITY = 0.1          # Opacity of the markers (0.0 transparent through 1.0 opaque)
COLORMAP = 'tab10'             # Colormap for clusters (e.g., 'tab10', 'tab20', etc.)
FIGURE_SIZE = (10, 8)         # Size of the figure in inches (width, height)
PLOT_TITLE = 't-SNE Visualization by Cluster'  # Title of the plot
X_LABEL = 't-SNE Dimension 1'       # Label for the X-axis
Y_LABEL = 't-SNE Dimension 2'       # Label for the Y-axis
GRID_ENABLED = False          # Whether to display grid lines
BACKGROUND_COLOR = 'white'    # Background color of the plot
ANNOTATION = False            # Whether to annotate points with TimeStamp
ANNOTATION_FONT_SIZE = 8      # Font size for annotations (if enabled)
ANNOTATION_COLOR = 'black'    # Color for annotations (if enabled)
LEGEND_TITLE = 'Cluster'      # Title for the legend
LEGEND_FONT_SIZE = 12         # Font size for legend text
# ============================ END OF CONFIGURABLE CONSTANTS =====================

def parse_arguments():
    parser = argparse.ArgumentParser(
        description="Plot t-SNE results from a CSV file containing TimeStamp, log, X, Y, and cluster columns."
    )
    parser.add_argument(
        "input_csv",
        type=str,
        help="Path to the input CSV file containing TimeStamp, log, X, Y, and cluster columns."
    )
    parser.add_argument(
        "-o", "--output",
        type=str,
        default="tsne_plot.png",
        help="Path to save the output plot image (default: tsne_plot.png)"
    )
    parser.add_argument(
        "-s", "--show",
        action='store_true',
        help="Display the plot interactively after saving."
    )
    return parser.parse_args()

def load_data(csv_path):
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
    required_columns = {'TimeStamp', 'log', 'X', 'Y', 'cluster'}
    if not required_columns.issubset(df.columns):
        missing = required_columns - set(df.columns)
        print(f"Error: The following required columns are missing in the CSV: {missing}")
        sys.exit(1)
    
    return df[['TimeStamp', 'log', 'X', 'Y', 'cluster']]

def plot_tsne(df, output_path, show_plot=False):
    plt.figure(figsize=FIGURE_SIZE)
    
    # Ensure 'cluster' is treated as a categorical variable
    df['cluster'] = df['cluster'].astype(str)
    unique_clusters = sorted(df['cluster'].unique())
    n_clusters = len(unique_clusters)
    
    # Define the colormap
    cmap = plt.get_cmap(COLORMAP)
    colors = cmap.colors if hasattr(cmap, 'colors') else cmap(range(n_clusters))
    
    # Handle cases where the number of clusters exceeds the colormap
    if n_clusters > len(colors):
        # Generate a list of colors by cycling through the colormap
        colors = [cmap(i % len(colors)) for i in range(n_clusters)]
        print(f"Warning: Number of clusters ({n_clusters}) exceeds colors in '{COLORMAP}' colormap. Colors will repeat.")
    
    # Create a mapping from cluster label to color
    cluster_to_color = {cluster: colors[idx] for idx, cluster in enumerate(unique_clusters)}
    
    # Assign colors to each point based on cluster
    point_colors = df['cluster'].map(cluster_to_color)
    
    scatter = plt.scatter(
        df['X'],
        df['Y'],
        s=MARKER_SIZE,
        alpha=MARKER_OPACITY,
        c=point_colors,
        edgecolors='w',
        linewidths=0.5
    )
    
    plt.title(PLOT_TITLE, fontsize=16)
    plt.xlabel(X_LABEL, fontsize=14)
    plt.ylabel(Y_LABEL, fontsize=14)
    
    if GRID_ENABLED:
        plt.grid(True, linestyle='--', alpha=0.5)
    
    plt.gca().set_facecolor(BACKGROUND_COLOR)
    
    # Create custom handles for the legend
    handles = [plt.Line2D([0], [0], marker='o', color='w',
                          markerfacecolor=cluster_to_color[cluster],
                          markersize=10, label=f'Cluster {cluster}')
               for cluster in unique_clusters]
    
    plt.legend(handles=handles, title=LEGEND_TITLE, fontsize=LEGEND_FONT_SIZE, title_fontsize=LEGEND_FONT_SIZE)
    
    # Optional: Annotate points with TimeStamp
    if ANNOTATION:
        for idx, row in df.iterrows():
            plt.annotate(
                str(row['TimeStamp']),
                (row['X'], row['Y']),
                textcoords="offset points",
                xytext=(0,5),
                ha='center',
                fontsize=ANNOTATION_FONT_SIZE,
                color=ANNOTATION_COLOR
            )
    
    plt.tight_layout()
    
    try:
        plt.savefig(output_path, dpi=300)
        print(f"Plot saved as '{output_path}'.")
    except Exception as e:
        print(f"Error saving the plot: {e}")
        sys.exit(1)
    
    if show_plot:
        plt.show()
    
    plt.close()

def main():
    args = parse_arguments()
    input_csv = args.input_csv
    output_plot = args.output
    show_plot = args.show
    
    # Load data
    df = load_data(input_csv)
    
    # Plot and save
    plot_tsne(df, output_plot, show_plot)

if __name__ == "__main__":
    main()
