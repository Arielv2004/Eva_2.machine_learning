import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score


# =====================================
# 🔹 1. PCA con columnas numéricas (CORREGIDO)
# =====================================
def apply_pca(data: pd.DataFrame, n_components: int):

    # Seleccionar solo columnas numéricas
    numeric_data = data.select_dtypes(include=['int64', 'float64'])

    # Eliminar NaN para evitar errores
    numeric_data = numeric_data.dropna()

    # Escalar datos
    scaler = StandardScaler()
    data_scaled = scaler.fit_transform(numeric_data)

    # Aplicar PCA
    pca = PCA(n_components=n_components, random_state=42)
    X_pca = pca.fit_transform(data_scaled)

    # ✔ Solo retornamos 2 outputs (solución 1)
    return pd.DataFrame(X_pca), pca.explained_variance_ratio_.tolist()



# =====================================
# 🔹 2. Ejecutar KMeans, DBSCAN y Jerárquico (CORREGIDO)
# =====================================
def run_clustering_algorithms(X_pca):

    results = {}

    # ⭐ KMEANS
    kmeans = KMeans(n_clusters=5, random_state=42)
    labels_kmeans = kmeans.fit_predict(X_pca)
    results["kmeans"] = {"labels": labels_kmeans.tolist()}

    # ⭐ DBSCAN
    try:
        db = DBSCAN(eps=0.5, min_samples=5)
        labels_db = db.fit_predict(X_pca)
        results["dbscan"] = {"labels": labels_db.tolist()}
    except Exception as e:
        results["dbscan"] = {"error": str(e)}

    # ⭐ AGGLOMERATIVE (CLUSTERING JERÁRQUICO)
    try:
        aggl = AgglomerativeClustering(
            n_clusters=5,
            linkage="ward",
            compute_full_tree=False
        )
        labels_aggl = aggl.fit_predict(X_pca)
        results["agglomerative"] = {"labels": labels_aggl.tolist()}  # ✔ nombre correcto
    except Exception as e:
        results["agglomerative"] = {"error": str(e)}

    return results



# =====================================
# 🔹 3. Agregar etiquetas de clusters (CORREGIDO)
# =====================================
def attach_cluster_labels(data: pd.DataFrame, results: dict):
    data_out = data.copy()

    # ✔ KMeans
    data_out["cluster_kmeans"] = results["kmeans"]["labels"]

    # ✔ DBSCAN
    data_out["cluster_dbscan"] = results["dbscan"]["labels"]

    # ❗ Usabas "hierarchical", pero el key real es "agglomerative"
    data_out["cluster_hierarchical"] = results["agglomerative"]["labels"]

    return data_out
