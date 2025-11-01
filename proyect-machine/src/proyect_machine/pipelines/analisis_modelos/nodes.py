import pandas as pd
import matplotlib.pyplot as plt

def comparar_modelos(
    linear_simple_metrics,
    linear_multiple_metrics,
    decision_tree_metrics,
    random_forest_metrics,
    knn_regressor_metrics,
):
    # Detectar automáticamente las llaves correctas
    def get_metric(metrics, *possible_keys):
        for key in possible_keys:
            if key in metrics:
                return metrics[key]
        return None

    data = {
        "Modelo": ["Lineal Simple", "Lineal Múltiple", "Árbol de Decisión", "Random Forest", "KNN"],
        "MSE": [
            get_metric(linear_simple_metrics, "mse", "mean_squared_error"),
            get_metric(linear_multiple_metrics, "mse", "mean_squared_error"),
            get_metric(decision_tree_metrics, "mse", "mean_squared_error"),
            get_metric(random_forest_metrics, "mse", "mean_squared_error"),
            get_metric(knn_regressor_metrics, "mse", "mean_squared_error"),
        ],
        "R2": [
            get_metric(linear_simple_metrics, "r2", "r2_score"),
            get_metric(linear_multiple_metrics, "r2", "r2_score"),
            get_metric(decision_tree_metrics, "r2", "r2_score"),
            get_metric(random_forest_metrics, "r2", "r2_score"),
            get_metric(knn_regressor_metrics, "r2", "r2_score"),
        ],
    }

    df = pd.DataFrame(data)
    print(df)

    # Crear gráfico
    fig, ax1 = plt.subplots(figsize=(8, 5))
    df.plot(x="Modelo", y="MSE", kind="bar", color="skyblue", ax=ax1)
    ax1.set_ylabel("Error Cuadrático Medio (MSE)")
    plt.title("Comparación de Modelos (MSE y R²)")

    # Añadir eje secundario para R²
    ax2 = ax1.twinx()
    df.plot(x="Modelo", y="R2", kind="line", marker="o", color="red", ax=ax2)
    ax2.set_ylabel("Coeficiente de Determinación (R²)")

    plt.tight_layout()
    plt.savefig("data/08_reporting/comparacion_modelos.png")
    plt.show()

    return df
