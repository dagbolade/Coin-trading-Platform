import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import pairwise_distances_argmin_min, pairwise_distances


def apply_kmeans(data, n_clusters=4, random_state=42):
    """
    Apply K-Means clustering to the dataset.

    Args:
        data (DataFrame or array-like): Data to cluster.
        n_clusters (int): Number of clusters to form.
        random_state (int): Random state for reproducibility.

    Returns:
        ndarray: Cluster labels.
    """
    kmeans = KMeans(n_clusters=n_clusters, random_state=random_state)
    clusters = kmeans.fit_predict(data)
    return clusters, kmeans


def add_cluster_labels(data, clusters):
    """
    Add cluster labels to the data.

    Args:
        data (DataFrame): Data to add labels to.
        clusters (ndarray): Cluster labels.

    Returns:
        DataFrame: Data with a new 'Cluster' column.
    """
    data['Cluster'] = clusters
    return data


def plot_clusters(data, clusters):
    """
    Visualize clusters in a scatter plot based on the first two principal components.

    Args:
        data (DataFrame or array-like): Data reduced by PCA.
        clusters (ndarray): Cluster labels.
    """
    # Create a figure and axis objects
    fig, ax = plt.subplots(figsize=(8, 5))

    # Use the axis object to plot data
    scatter = ax.scatter(data[:, 0], data[:, 1], c=clusters, cmap='viridis')

    # Add titles and labels
    ax.set_title('Cryptocurrency Clusters')
    ax.set_xlabel('Principal Component 1')
    ax.set_ylabel('Principal Component 2')

    # Create a colorbar
    colorbar = fig.colorbar(scatter, ax=ax)
    colorbar.set_label('Cluster')

    # Return the figure object to be used with Streamlit
    return fig


def plot_cluster_distribution(data):
    """
    Plot the distribution of points across clusters.

    Args:
        data (DataFrame): Data with a 'Cluster' column.
    """
    # Create a figure and axis objects
    fig, ax = plt.subplots(figsize=(10, 7))

    # Use seaborn's countplot with the axis object
    ax = sns.countplot(x='Cluster', data=data, palette='viridis', ax=ax)

    # Add titles and labels
    ax.set_title('Cryptocurrency Clusters')
    ax.set_xlabel('Cluster')
    ax.set_ylabel('Count')

    # Annotate counts above bars
    for p in ax.patches:
        ax.annotate(f'{int(p.get_height())}', (p.get_x() + p.get_width() / 2., p.get_height()),
                    ha='center', va='center', fontsize=11, color='black', xytext=(0, 5),
                    textcoords='offset points')

    # Return the figure object to be used with Streamlit
    return fig


def select_cryptos_closest_to_centroids(data, clusters, cluster_centers):
    distances = pairwise_distances(data, cluster_centers)

    closest_indices = np.argmin(distances, axis=0)

    selected = data.iloc[closest_indices]

    selected.loc[:, 'Cluster'] = clusters[closest_indices]
    return selected
