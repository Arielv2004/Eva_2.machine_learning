from kedro.pipeline import Pipeline, node
from .nodes import scale_data, apply_pca, apply_tsne


def create_pipeline(**kwargs):
    return Pipeline([
        # 1️⃣ Escalar los datos
        node(
            func=scale_data,
            inputs=dict(
                data="clean_movies",
                feature_cols="params:feature_cols2"
            ),
            outputs="scaled_movies",
            name="scale_data_node"
        ),

        # 2️⃣ PCA
        node(
            func=apply_pca,
            inputs=dict(
                X_scaled="scaled_movies",
                parameters="parameters"
            ),
            outputs=["X_pca", "pca_variance"],
            name="pca_node"
        ),

        # 3️⃣ t-SNE
        node(
            func=apply_tsne,
            inputs=dict(
                X_scaled="scaled_movies",
                parameters="parameters"
            ),
            outputs="X_tsne",
            name="tsne_node"
        )
    ])
