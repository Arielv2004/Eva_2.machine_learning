import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE


# Escalar los datos
def scale_data(data: pd.DataFrame, feature_cols: list[str]):
    X = data[feature_cols].dropna()
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    return pd.DataFrame(X_scaled, columns=feature_cols)

# PCA SEGURO
def apply_pca(X_scaled: pd.DataFrame, parameters: dict):
    n_components = parameters["pca"]["n_components"]

    max_components = min(X_scaled.shape[1], n_components)

    pca = PCA(n_components=max_components)
    X_pca = pca.fit_transform(X_scaled)

    explained_var = pd.DataFrame({
        "component": [f"PC{i+1}" for i in range(max_components)],
        "explained_variance": pca.explained_variance_ratio_
    })

    return pd.DataFrame(X_pca, columns=[f"PC{i+1}" for i in range(max_components)]), explained_var

# t-SNE
def apply_tsne(X_scaled: pd.DataFrame, parameters: dict):
    tsne_params = parameters["tsne"]

    tsne = TSNE(
        n_components=tsne_params["n_components"],
        perplexity=tsne_params["perplexity"],
        learning_rate="auto",
        init="random"
    )

    X_tsne = tsne.fit_transform(X_scaled)
    return pd.DataFrame(X_tsne, columns=["TSNE1", "TSNE2"])
