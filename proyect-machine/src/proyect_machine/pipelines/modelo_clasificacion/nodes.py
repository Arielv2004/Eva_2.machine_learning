import os
import pickle
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report

# --- Función auxiliar para guardar modelos y métricas ---
def guardar_modelo_y_metricas(nombre, modelo, y_test, y_pred):
    os.makedirs("data/06_models", exist_ok=True)
    with open(f"data/06_models/{nombre}_model.pkl", "wb") as f:
        pickle.dump(modelo, f)
    metricas = {
        "accuracy": accuracy_score(y_test, y_pred),
        "report": classification_report(y_test, y_pred, output_dict=True)
    }
    with open(f"data/06_models/{nombre}_metrics.pkl", "wb") as f:
        pickle.dump(metricas, f)
    print(f" Modelo y métricas guardadas: {nombre}")

# --- Logistic Regression ---
def train_logistic_regression(movies_metadata: pd.DataFrame):
    movies_metadata['high_rating'] = (movies_metadata['vote_average'] > 6).astype(int)
    numeric_cols = ['vote_count', 'popularity', 'runtime', 'budget']
    movies_metadata[numeric_cols] = movies_metadata[numeric_cols].apply(pd.to_numeric, errors='coerce')
    movies_metadata = movies_metadata.dropna(subset=numeric_cols + ['high_rating'])

    X = movies_metadata[numeric_cols]
    y = movies_metadata['high_rating']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=42)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    model = LogisticRegression(class_weight='balanced', max_iter=1000)
    model.fit(X_train_scaled, y_train)
    y_pred = model.predict(X_test_scaled)

    print("Logistic Regression Accuracy:", accuracy_score(y_test, y_pred))
    print(classification_report(y_test, y_pred))

    guardar_modelo_y_metricas("logistic_regression_class", model, y_test, y_pred)
    return model

# --- Decision Tree ---
def train_decision_tree(movies_metadata: pd.DataFrame):
    movies_metadata['high_rating'] = (movies_metadata['vote_average'] > 6).astype(int)
    numeric_cols = ['runtime', 'popularity', 'vote_count', 'revenue', 'budget']
    movies_metadata[numeric_cols] = movies_metadata[numeric_cols].apply(pd.to_numeric, errors='coerce')
    movies_metadata = movies_metadata.dropna(subset=numeric_cols + ['high_rating'])

    X = movies_metadata[numeric_cols]
    y = movies_metadata['high_rating']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=42)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    model = DecisionTreeClassifier(criterion='gini', max_depth=10, class_weight='balanced', random_state=42)
    model.fit(X_train_scaled, y_train)
    y_pred = model.predict(X_test_scaled)

    print("Decision Tree Accuracy:", accuracy_score(y_test, y_pred))
    print(classification_report(y_test, y_pred))

    guardar_modelo_y_metricas("decision_tree_class", model, y_test, y_pred)
    return model

# --- KNN ---
def train_knn(movies_metadata: pd.DataFrame):
    movies_metadata['high_rating'] = (movies_metadata['vote_average'] > 6).astype(int)
    selected_features = ['runtime', 'popularity', 'vote_count', 'revenue', 'budget']
    movies_metadata[selected_features] = movies_metadata[selected_features].apply(pd.to_numeric, errors='coerce')
    movies_metadata = movies_metadata.dropna(subset=selected_features + ['high_rating'])

    X = movies_metadata[selected_features]
    y = movies_metadata['high_rating']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=42)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    model = KNeighborsClassifier(n_neighbors=160, weights='distance', metric='minkowski', p=2)
    model.fit(X_train_scaled, y_train)
    y_pred = model.predict(X_test_scaled)

    print("KNN Accuracy:", accuracy_score(y_test, y_pred))
    print(classification_report(y_test, y_pred))

    guardar_modelo_y_metricas("knn_class", model, y_test, y_pred)
    return model

# --- Random Forest ---
def train_random_forest(movies_metadata: pd.DataFrame):
    movies_metadata['high_rating'] = (movies_metadata['vote_average'] > 6).astype(int)
    selected_features = ['runtime', 'popularity', 'vote_count', 'revenue', 'budget']
    movies_metadata[selected_features] = movies_metadata[selected_features].apply(pd.to_numeric, errors='coerce')
    movies_metadata = movies_metadata.dropna(subset=selected_features + ['high_rating'])

    X = movies_metadata[selected_features]
    y = movies_metadata['high_rating']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=42)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    model = RandomForestClassifier(
        n_estimators=200,
        max_depth=10,
        min_samples_split=5,
        min_samples_leaf=2,
        class_weight='balanced',
        random_state=42
    )
    model.fit(X_train_scaled, y_train)
    y_pred = model.predict(X_test_scaled)

    print("Random Forest Accuracy:", accuracy_score(y_test, y_pred))
    print(classification_report(y_test, y_pred))

    guardar_modelo_y_metricas("random_forest_class", model, y_test, y_pred)
    return model

# --- SVM ---
def train_svm(movies_metadata: pd.DataFrame):
    movies_metadata['high_rating'] = (movies_metadata['vote_average'] > 6).astype(int)
    selected_features = ['runtime', 'popularity', 'vote_count', 'revenue', 'budget']
    movies_metadata[selected_features] = movies_metadata[selected_features].apply(pd.to_numeric, errors='coerce')
    movies_metadata = movies_metadata.dropna(subset=selected_features + ['high_rating'])

    X = movies_metadata[selected_features]
    y = movies_metadata['high_rating']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=42)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    model = SVC(kernel='rbf', class_weight='balanced', random_state=42)
    model.fit(X_train_scaled, y_train)
    y_pred = model.predict(X_test_scaled)

    print("SVM Accuracy:", accuracy_score(y_test, y_pred))
    print(classification_report(y_test, y_pred))

    guardar_modelo_y_metricas("svm_class", model, y_test, y_pred)
    return model
