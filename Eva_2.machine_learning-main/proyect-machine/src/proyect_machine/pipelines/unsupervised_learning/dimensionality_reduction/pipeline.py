from kedro.pipeline import Pipeline, node
from .nodes import run_pca, apply_tsne

def create_pipeline(**kwargs):
    return Pipeline([
        node(
            func=run_pca,
            inputs=dict(data="movies_metadata", parameters="parameters"),
            outputs="X_pca",  # Coincide con catalog.yml
            name="pca_node",
        ),
        node(
            func=apply_tsne,
            inputs=dict(data="X_pca", parameters="parameters"),  # Usamos la salida de PCA
            outputs="X_tsne",  # Este dataset debes definir en catalog.yml
            name="apply_tsne_node"
        )
    ])
