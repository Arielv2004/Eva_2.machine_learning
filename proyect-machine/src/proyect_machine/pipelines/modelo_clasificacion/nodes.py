import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report

def train_model(data: pd.DataFrame):
    # Crear la variable objetivo
    data['high_rating'] = (data['vote_average'] > 6).astype(int)
    
    # Seleccionar columnas
    numeric_cols = ['vote_count', 'popularity', 'runtime', 'budget']
    data[numeric_cols] = data[numeric_cols].apply(pd.to_numeric, errors='coerce')
    data = data.dropna(subset=numeric_cols + ['high_rating'])
    
    # Definir X e y
    X = data[numeric_cols]
    y = data['high_rating']
    
    # División train/test
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=42)
    
    # Escalado
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Entrenar modelo SVC
    model = SVC(probability=True, kernel='linear', class_weight='balanced', random_state=42)
    model.fit(X_train_scaled, y_train)
    
    # Predicciones
    y_pred = model.predict(X_test_scaled)
    
    # Métricas
    acc = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred)
    
    print("Accuracy:", acc)
    print("\nReporte de clasificación:\n", report)
    
    return model
