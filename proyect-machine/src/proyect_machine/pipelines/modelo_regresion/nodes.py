import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np

# Función auxiliar general

def train_and_evaluate_model(model, X_train, X_test, y_train, y_test):
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    return model, {"MSE": mse, "R2": r2}


# Regresión Lineal Simple

def train_linear_simple(movies_metadata: pd.DataFrame):
    data = movies_metadata.dropna(subset=["budget", "revenue"])
    X = data[["budget"]]
    y = data["revenue"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = LinearRegression()

    return train_and_evaluate_model(model, X_train, X_test, y_train, y_test)


# Regresión Lineal Múltiple

def train_linear_multiple(movies_metadata: pd.DataFrame):
    data = movies_metadata.dropna(subset=["budget", "popularity", "runtime", "revenue"])
    X = data[["budget", "popularity", "runtime"]]
    y = data["revenue"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = LinearRegression()

    return train_and_evaluate_model(model, X_train, X_test, y_train, y_test)


# Árbol de Decisión

def train_decision_tree(movies_metadata: pd.DataFrame):
    data = movies_metadata.dropna(subset=["budget", "popularity", "runtime", "revenue"])
    X = data[["budget", "popularity", "runtime"]]
    y = data["revenue"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = DecisionTreeRegressor(random_state=42)

    return train_and_evaluate_model(model, X_train, X_test, y_train, y_test)

# Random Forest

def train_random_forest(movies_metadata: pd.DataFrame):
    data = movies_metadata.dropna(subset=["budget", "popularity", "runtime", "revenue"])
    X = data[["budget", "popularity", "runtime"]]
    y = data["revenue"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = RandomForestRegressor(n_estimators=100, random_state=42)

    return train_and_evaluate_model(model, X_train, X_test, y_train, y_test)

# KNN Regressor

def train_knn_regressor(movies_metadata: pd.DataFrame):
    data = movies_metadata.dropna(subset=["budget", "popularity", "runtime", "revenue"])
    X = data[["budget", "popularity", "runtime"]]
    y = data["revenue"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    model = KNeighborsRegressor(n_neighbors=5)

    return train_and_evaluate_model(model, X_train_scaled, X_test_scaled, y_train, y_test)
