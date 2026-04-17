from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from beartype import beartype
from jaxtyping import Float, Shaped, jaxtyped
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from scipy.stats import chi2
from scipy.spatial.distance import mahalanobis


@jaxtyped(typechecker=beartype)
def plot_outliers(
    data: Float[np.ndarray, "n_samples 3"],
    mahalanobis_dist: Float[np.ndarray, "n_samples"],
    save_path: str | Path,
) -> None:
    """Plot a 3D scatter of reduced-dimension data colored by Mahalanobis distance.

    Args:
        data: Array of shape (n_samples, 3) containing the 3D t-SNE coordinates to plot.
        mahalanobis_dist: Per-sample Mahalanobis distances used to color the scatter points.
        save_path: File path where the plot image will be saved.
    """
    norm = plt.Normalize(mahalanobis_dist.min(), mahalanobis_dist.max())
    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(111, projection="3d")
    sc = ax.scatter(
        data[:, 0], data[:, 1], data[:, 2],
        c=mahalanobis_dist,
        cmap="coolwarm",
        norm=norm,
        label="Mahalanobis distances",
    )
    ax.set_title("Mahalanobis Distance (PCA + t-SNE Components)")
    ax.set_xlabel("Component 1")
    ax.set_ylabel("Component 2")
    ax.set_zlabel("Component 3")
    plt.colorbar(sc, ax=ax)
    plt.legend()
    plt.savefig(save_path)
    plt.close()


@jaxtyped(typechecker=beartype)
def calculate_mahal(
    data: Float[np.ndarray, "n_samples n_features"],
    y: Shaped[np.ndarray, "n_samples"],
    alpha: float = 0.05,
) -> tuple[Float[np.ndarray, "n_samples"], Float[np.ndarray, "n_samples"]]:
    """Compute per-class Mahalanobis distances and flag outliers via a chi-squared threshold.

    Args:
        data: Array of shape (n_samples, n_features) containing the input data.
        y: Array of shape (n_samples,) with class labels for each sample.
        alpha: Significance level for the chi-squared outlier threshold.

    Returns:
        A tuple of (distances, outliers) where distances holds the per-sample Mahalanobis
        distance and outliers is a binary array (1 = outlier, 0 = inlier).
    """
    total_distances = np.zeros(len(y), dtype=np.float64)
    total_outliers = np.zeros(len(y), dtype=np.float64)
    threshold = np.sqrt(chi2.ppf(1 - alpha, df=data.shape[1]))

    for cls in np.unique(y):
        mask = y == cls
        sub = data[mask]

        mean = sub.mean(axis=0)
        try:
            inv_cov = np.linalg.inv(np.cov(sub.T))
        except np.linalg.LinAlgError:
            print(f"Warning: singular covariance matrix for class {cls!r}, skipping.")
            continue

        dists = np.array([mahalanobis(x, mean, inv_cov) for x in sub])
        total_distances[mask] = dists
        total_outliers[mask] = (dists > threshold).astype(np.float64)

    return total_distances, total_outliers


@jaxtyped(typechecker=beartype)
def reduce_pca(
    data: Float[np.ndarray, "n_samples n_features"],
    n_components: int = 15,
) -> Float[np.ndarray, "n_samples n_components"]:
    """Reduce data dimensionality using PCA.

    Args:
        data: Input array of shape (n_samples, n_features).
        n_components: Number of principal components to retain.

    Returns:
        Transformed array of shape (n_samples, n_components).
    """
    return PCA(n_components=n_components).fit_transform(data)


@jaxtyped(typechecker=beartype)
def reduce_tsne(
    data: Float[np.ndarray, "n_samples n_features"],
    n_components: int = 3,
) -> Float[np.ndarray, "n_samples n_components"]:
    """Reduce data dimensionality using t-SNE.

    Args:
        data: Input array of shape (n_samples, n_features).
        n_components: Number of t-SNE embedding dimensions.

    Returns:
        Embedded array of shape (n_samples, n_components).
    """
    return TSNE(n_components=n_components, random_state=42).fit_transform(data)


def load_data(csv_path: str | Path) -> tuple[np.ndarray, np.ndarray]:
    """Load and preprocess the dataset, returning features and labels as numpy arrays.

    Args:
        csv_path: Path to the CSV file containing the dataset.

    Returns:
        A tuple of (features, labels) where features is a float64 array of shape
        (n_samples, n_features) and labels is an object array of shape (n_samples,).

    Raises:
        FileNotFoundError: If csv_path does not exist.
    """
    path = Path(csv_path)
    if not path.exists():
        raise FileNotFoundError(f"Data file not found: {path}")

    df = pd.read_csv(path)
    labels = df["diagnosis"].to_numpy()
    features = (
        df.drop(columns=["id", "diagnosis"])
        .dropna(axis=1)
        .to_numpy(dtype=np.float64)
    )
    return features, labels


def main() -> None:
    """Run the full outlier detection pipeline."""
    features, labels = load_data("data.csv")

    data_pca = reduce_pca(features)
    data_tsne = reduce_tsne(data_pca)

    distances, _ = calculate_mahal(data_pca, labels)

    plot_outliers(data_tsne, distances, "outliers.png")
    print("Saved outliers.png")


if __name__ == "__main__":
    main()
