from kedro.pipeline import Pipeline, node
from .nodes import (
    clean_data,
    run_clustering_algorithms,
    attach_cluster_labels
)

def create_pipeline(**kwargs):
    return Pipeline([
        
        # Limpiar datos
        node(
            func=clean_data,
            inputs="movies_metadata",
            outputs="clean_movies",
            name="clean_movies_node"
        ),

        # Clustering 
        node(
            func=run_clustering_algorithms,
            inputs=["X_pca", "parameters"],
            outputs="cluster_results",
            name="clustering_algorithms_node"
        ),

        # Unir clusters al dataset original
        node(
            func=attach_cluster_labels,
            inputs=["clean_movies", "cluster_results"],
            outputs="movies_with_clusters",
            name="attach_labels_node"
        ),
    ])
