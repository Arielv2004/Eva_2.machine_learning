import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.impute import SimpleImputer

# Selecciona columnas numéricas
def select_numeric_columns(data: pd.DataFrame, feature_cols: list) -> pd.DataFrame:
    return data[feature_cols].select_dtypes(include=['float64', 'int64'])

# Función genérica para entrenar modelos
def train_model(data: pd.DataFrame, target_col: str, feature_cols: list, test_size: float, random_state: int, model_type='linear', impute_X=True):
    X = select_numeric_columns(data, feature_cols)
    y = data[target_col]

    # Quitar filas donde y sea NaN
    mask = y.notna()
    X = X[mask]
    y = y[mask]

    # Imputar NaN en X si se indica
    if impute_X:
        imputer = SimpleImputer(strategy='mean')
        X = pd.DataFrame(imputer.fit_transform(X), columns=X.columns, index=X.index)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)

    if model_type == 'linear':
        model = LinearRegression()
    elif model_type == 'decision_tree':
        model = DecisionTreeRegressor(random_state=random_state)
    elif model_type == 'random_forest':
        model = RandomForestRegressor(random_state=random_state)
    else:
        raise ValueError(f"Tipo de modelo no soportado: {model_type}")

    model.fit(X_train, y_train)
    preds = model.predict(X_test)

    metrics = {
        "mse": mean_squared_error(y_test, preds),
        "r2": r2_score(y_test, preds)
    }

    return model, metrics

# Funciones para Kedro
def train_linear_regression_imputer(data, target_col, feature_cols, test_size, random_state):
    return train_model(data, target_col, feature_cols, test_size, random_state, model_type='linear', impute_X=True)

def train_linear_regression_dropna(data, target_col, feature_cols, test_size, random_state):
    return train_model(data, target_col, feature_cols, test_size, random_state, model_type='linear', impute_X=False)

def train_multiple_linear_regression(data, target_col, feature_cols, test_size, random_state):
    # Aquí ahora también imputamos X
    return train_model(data, target_col, feature_cols, test_size, random_state, model_type='linear', impute_X=True)

def train_decision_tree(data, target_col, feature_cols, test_size, random_state):
    return train_model(data, target_col, feature_cols, test_size, random_state, model_type='decision_tree', impute_X=True)

def train_random_forest(data, target_col, feature_cols, test_size, random_state):
    return train_model(data, target_col, feature_cols, test_size, random_state, model_type='random_forest', impute_X=True)
