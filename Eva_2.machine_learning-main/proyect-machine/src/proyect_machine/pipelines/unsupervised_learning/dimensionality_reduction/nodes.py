import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.manifold import TSNE

# 🔹 Node PCA
from sklearn.decomposition import PCA

def run_pca(data, parameters):
    n_components = parameters["pca"]["n_components"]
    pca = PCA(n_components=n_components)
    transformed = pca.fit_transform(data)
    return transformed


# 🔹 Node t-SNE
def apply_tsne(data: pd.DataFrame, n_components: int = 2, perplexity: int = 30, random_state: int = 42):
    tsne = TSNE(n_components=n_components, perplexity=perplexity, random_state=random_state)
    X_tsne = tsne.fit_transform(data)
    return pd.DataFrame(X_tsne)
