import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_squared_error, r2_score
import pickle, os

# Función auxiliar para guardar modelos y métricas 
def guardar_modelo_y_metricas_regresion(nombre, modelo, y_test, y_pred):
    ruta_guardado = "./04_modelos_regresion/data/06_models"
    os.makedirs(ruta_guardado, exist_ok=True)
    
    # Guardar modelo
    with open(f"{ruta_guardado}/{nombre}_model.pkl", "wb") as f:
        pickle.dump(modelo, f)
    
    # Calcular métricas
    r2 = r2_score(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    metricas = {"R2": r2, "MSE": mse}
    
    # Guardar métricas
    with open(f"{ruta_guardado}/{nombre}_metrics.pkl", "wb") as f:
        pickle.dump(metricas, f)
    
    # Mostrar métricas en consola
    print(f" {nombre} entrenado y guardado")
    print(f"R2: {r2:.4f}, MSE: {mse:.4f}\n")
    
    return metricas


def train_and_evaluate_model(model, X_train, X_test, y_train, y_test, nombre_modelo):
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    metricas = guardar_modelo_y_metricas_regresion(nombre_modelo, model, y_test, y_pred)
    return model, metricas

# --- Modelos de regresión ---

def train_linear_simple(movies_metadata: pd.DataFrame):
    data = movies_metadata.dropna(subset=["budget", "revenue"])
    X = data[["budget"]]
    y = data["revenue"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = LinearRegression()

    return train_and_evaluate_model(model, X_train, X_test, y_train, y_test, "linear_simple")

def train_linear_multiple(movies_metadata: pd.DataFrame):
    data = movies_metadata.dropna(subset=["budget", "popularity", "runtime", "revenue"])
    X = data[["budget", "popularity", "runtime"]]
    y = data["revenue"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = LinearRegression()

    return train_and_evaluate_model(model, X_train, X_test, y_train, y_test, "linear_multiple")

def train_decision_tree(movies_metadata: pd.DataFrame):
    data = movies_metadata.dropna(subset=["budget", "popularity", "runtime", "revenue"])
    X = data[["budget", "popularity", "runtime"]]
    y = data["revenue"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = DecisionTreeRegressor(random_state=42)

    return train_and_evaluate_model(model, X_train, X_test, y_train, y_test, "decision_tree")

def train_random_forest(movies_metadata: pd.DataFrame):
    data = movies_metadata.dropna(subset=["budget", "popularity", "runtime", "revenue"])
    X = data[["budget", "popularity", "runtime"]]
    y = data["revenue"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = RandomForestRegressor(n_estimators=100, random_state=42)

    return train_and_evaluate_model(model, X_train, X_test, y_train, y_test, "random_forest")

def train_knn_regressor(movies_metadata: pd.DataFrame):
    data = movies_metadata.dropna(subset=["budget", "popularity", "runtime", "revenue"])
    X = data[["budget", "popularity", "runtime"]]
    y = data["revenue"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    model = KNeighborsRegressor(n_neighbors=5)

    return train_and_evaluate_model(model, X_train_scaled, X_test_scaled, y_train, y_test, "knn_regressor")
