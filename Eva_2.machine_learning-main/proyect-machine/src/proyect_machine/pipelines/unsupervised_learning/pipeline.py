from kedro.pipeline import Pipeline, node
from .nodes import apply_pca, run_clustering_algorithms, attach_cluster_labels


def create_pipeline(**kwargs):
    return Pipeline(
        [
            node(
                func=apply_pca,
                inputs=["movies_metadata", "params:n_components"],
                outputs=["X_pca", "pca_variance"],
                name="apply_pca_node"
            ),
            node(
                func=run_clustering_algorithms,
                inputs="X_pca",
                outputs="cluster_results",
                name="clustering_algorithms_node"
            ),
            node(
                func=attach_cluster_labels,
                inputs=["movies_metadata", "cluster_results"],
                outputs="movies_with_clusters",
                name="attach_labels_node"
            ),
        ]
    )
