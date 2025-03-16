import numpy as np
from matplotlib import pyplot as plt
from sklearn.decomposition import PCA
import pandas as pd


def apply_pca(df, n_components=10):
    """
    Apply PCA to reduce the dimensions of the dataset.

    Args:
        df: DataFrame, the input data to be reduced.
        n_components: int, the number of principal components to compute.

    Returns:
        DataFrame of the PCA-reduced data.
    """
    pca = PCA(n_components=n_components)
    reduced_data = pca.fit_transform(df)
    reduced_data_df = pd.DataFrame(reduced_data, index=df.index)
    return reduced_data_df, pca


def plot_explained_variance(pca, n_components=10):
    explained_variance = pca.explained_variance_ratio_
    indices = np.arange(n_components)
    values = explained_variance[:n_components]
    cumulative_values = np.cumsum(values)

    # Create a figure and axis to plot
    fig, ax = plt.subplots(figsize=(10, 6))

    # Use the axis `ax` for plotting operations
    bar = ax.bar(indices, values, alpha=0.6, label='Explained Variance', color='c')
    line, = ax.plot(indices, cumulative_values, label='Cumulative Explained Variance', marker='o', color='orange',
                    linestyle='-', linewidth=2)

    for i, (bar, value) in enumerate(zip(bar, values)):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(), f"{value:.2%}", ha='center', va='bottom')
        if i < len(cumulative_values):
            ax.text(indices[i], cumulative_values[i] + 0.02, f"{cumulative_values[i]:.2%}", ha='center', va='bottom')

    ax.set_xlabel('Principal Component')
    ax.set_ylabel('Explained Variance')
    ax.set_xticks(indices)
    ax.legend(loc='best')
    ax.set_title('Explained Variance of Principal Components')

    # Return the figure object to be used with Streamlit
    return fig
