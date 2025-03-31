import pdb
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import numpy as np
from scipy.stats import chi2
from scipy.spatial.distance import mahalanobis


# -----------------------------------------------------------------------------------------------
# Plot function
# -----------------------------------------------------------------------------------------------
def plot_outliers(data, mahalanobis_dist, save_name):
    norm = plt.Normalize(min(mahalanobis_dist), max(mahalanobis_dist))
    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(111, projection='3d')
    sc = ax.scatter(data[:, 0], data[:, 1], data[:, 2], c=mahalanobis_dist, cmap="coolwarm", norm=norm, label='Mahalanobis distances')
    ax.set_title("Mahalanobis Distance (PCA-TSNE Components)")
    ax.set_xlabel('Component 1')
    ax.set_ylabel('Component 2')
    ax.set_zlabel('Component 3')
    plt.legend()
    cbar = plt.colorbar(sc, ax=ax)
    plt.savefig(save_name)
    plt.close()


# -----------------------------------------------------------------------------------------------
# Calculate
# -----------------------------------------------------------------------------------------------
def calculate_mahal(data, y, alpha=0.05):
    total_outliers = np.zeros(len(y))  # Outlier flags (0 = not outlier, 1 = outlier)
    total_distances = np.zeros(len(y))  # Mahalanobis distances

    unique_classes = np.unique(y)  # Get unique classes

    for unique_class in unique_classes:
        sub_data = data[y == unique_class]

        mean = np.mean(sub_data, axis=0)
        try:
            cov = np.cov(sub_data.T)
            inv_cov = np.linalg.inv(cov)
        except np.linalg.LinAlgError:
            print(f"Warning: Singular covariance matrix for class {unique_class}. Skipping outlier detection.")
            continue

        mahalanobis_dist = np.array([mahalanobis(x, mean, inv_cov) for x in sub_data])

        C = np.sqrt(chi2.ppf(1 - alpha, df=sub_data.shape[1]))  # Cut-off threshold

        outliers = (mahalanobis_dist > C).astype(int)

        total_outliers[y == unique_class] = outliers
        total_distances[y == unique_class] = mahalanobis_dist

    return total_distances, total_outliers


# -----------------------------------------------------------------------------------------------
# Data dimensionality reduction
# -----------------------------------------------------------------------------------------------
def pca_data(data, n_components=15):
    pca = PCA(n_components=n_components)
    data_pca = pca.fit_transform(data)
    return data_pca

def tsne_data(data, n_components=3):
    tsne = TSNE(n_components=n_components, random_state=42)
    data_3d = tsne.fit_transform(data)
    return data_3d

# -----------------------------------------------------------------------------------------------
# Dummy data (Kaggle Cancer)
# -----------------------------------------------------------------------------------------------
df = pd.read_csv("data.csv")
print(df.head(2))
print(df.columns)
print(df.shape)


# -----------------------------------------------------------------------------------------------
# Format data
# -----------------------------------------------------------------------------------------------
labels = df["diagnosis"]
data = df.drop(['id', 'diagnosis'], axis=1)
data.dropna(axis=1, inplace=True)
print(labels.shape)
print(data.shape)

# -----------------------------------------------------------------------------------------------
# Dimensionality reduction: Calculate PCA and TSNE Components
# -----------------------------------------------------------------------------------------------
data_pca = pca_data(data)
data_tsne = tsne_data(data_pca)

# -----------------------------------------------------------------------------------------------
# Calculate Mahalanobis Distance
# -----------------------------------------------------------------------------------------------
mahalanobis_dist, outliers = calculate_mahal(data_pca, labels)

# -----------------------------------------------------------------------------------------------
# Plot 3-D
# -----------------------------------------------------------------------------------------------
plot_outliers(data_tsne, mahalanobis_dist, "outliers.png")
