import pandas as pd
import pickle

# Selecciona columnas numéricas
def select_numeric_columns(data: pd.DataFrame, feature_cols: list) -> pd.DataFrame:
    return data[feature_cols].select_dtypes(include=['float64', 'int64'])

# Node para cargar modelo desde pickle
def load_linear_model(model_path: str):
    """Carga un modelo de regresión lineal entrenado desde un archivo pickle."""
    with open(model_path, "rb") as f:
        model = pickle.load(f)
    return model

# Node para hacer predicciones usando el modelo cargado
def predict_with_model(model, data: pd.DataFrame, feature_cols: list, impute_X=True):
    """Recibe un modelo y datos, devuelve predicciones agregadas al DataFrame."""
    X = select_numeric_columns(data, feature_cols)

    # Imputar NaN si corresponde
    if impute_X:
        from sklearn.impute import SimpleImputer
        imputer = SimpleImputer(strategy='mean')
        X = pd.DataFrame(imputer.fit_transform(X), columns=X.columns, index=X.index)

    preds = model.predict(X)
    data = data.copy()
    data['predicted'] = preds
    return data
