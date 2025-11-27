from kedro.pipeline import Pipeline, node
from .nodes import clean_data, apply_pca, run_clustering_algorithms, attach_cluster_labels


def create_pipeline(**kwargs):
    return Pipeline(
        [
            node(
                func=clean_data,
                inputs="movies_metadata",
                outputs="clean_movies",
                name="clean_movies_node"
            ),

            node(
                func=apply_pca,
                inputs=["clean_movies", "params:n_components"],
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
                inputs=["clean_movies", "cluster_results"],
                outputs="movies_with_clusters",
                name="attach_labels_node"
            ),
        ]
    )
