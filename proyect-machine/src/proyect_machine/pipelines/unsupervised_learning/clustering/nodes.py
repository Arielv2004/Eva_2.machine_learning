import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans, DBSCAN
from sklearn.mixture import GaussianMixture
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score


# LIMPIAR DATA
def clean_data(data: pd.DataFrame) -> pd.DataFrame:
    numeric_data = data.select_dtypes(include=['float64', 'int64'])
    clean_df = numeric_data.fillna(numeric_data.mean())
    return clean_df


# PCA
def apply_pca(data: pd.DataFrame, n_components: int):
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(data)

    pca = PCA(n_components=n_components)
    X_pca = pca.fit_transform(scaled_data)

    pca_variance = pd.DataFrame({
        "component": list(range(1, n_components + 1)),
        "explained_variance_ratio": pca.explained_variance_ratio_
    })

    return pd.DataFrame(X_pca), pca_variance

def run_clustering_algorithms(X_pca: pd.DataFrame, parameters: dict):
    results = {}

    # KMEANS
    kmeans = KMeans(
        n_clusters=parameters["kmeans"]["clusters"],   # ← CAMBIO AQUÍ
        random_state=42
    )
    labels_kmeans = kmeans.fit_predict(X_pca)

    results["kmeans"] = {
        "labels": labels_kmeans.tolist(),
        "inertia": float(kmeans.inertia_),
        "silhouette": float(silhouette_score(X_pca, labels_kmeans)),
        "davies_bouldin": float(davies_bouldin_score(X_pca, labels_kmeans)),
        "calinski_harabasz": float(calinski_harabasz_score(X_pca, labels_kmeans))
    }

    # DBSCAN 
    dbscan = DBSCAN(
        eps=parameters["dbscan"]["eps"],                # ← CAMBIO AQUÍ
        min_samples=parameters["dbscan"]["min_samples"] # ← CAMBIO AQUÍ
    )
    labels_dbscan = dbscan.fit_predict(X_pca)

    results["dbscan"] = {
        "labels": labels_dbscan.tolist(),
        "silhouette": float(silhouette_score(X_pca, labels_dbscan)) if len(set(labels_dbscan)) > 1 else None,
        "davies_bouldin": float(davies_bouldin_score(X_pca, labels_dbscan)) if len(set(labels_dbscan)) > 1 else None,
        "calinski_harabasz": float(calinski_harabasz_score(X_pca, labels_dbscan)) if len(set(labels_dbscan)) > 1 else None
    }

    # GMM 
    gmm = GaussianMixture(
        n_components=parameters["gmm"]["components"],   # ← CAMBIO AQUÍ
        random_state=42
    )
    labels_gmm = gmm.fit_predict(X_pca)

    results["gmm"] = {
        "labels": labels_gmm.tolist(),
        "silhouette": float(silhouette_score(X_pca, labels_gmm)),
        "davies_bouldin": float(davies_bouldin_score(X_pca, labels_gmm)),
        "calinski_harabasz": float(calinski_harabasz_score(X_pca, labels_gmm))
    }

    return results


# UNIR CLUSTERS AL DATASET

def attach_cluster_labels(original_df: pd.DataFrame, cluster_results: dict):
    df = original_df.copy()

    df["cluster_kmeans"] = cluster_results["kmeans"]["labels"]
    df["cluster_dbscan"] = cluster_results["dbscan"]["labels"]
    df["cluster_gmm"] = cluster_results["gmm"]["labels"]

    return df
