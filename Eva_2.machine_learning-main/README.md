#  Proyecto Machine Learning – Análisis y Modelos Predictivos (CRISP-DM)

[![Python](https://img.shields.io/badge/Python-3.12-blue.svg)](https://www.python.org/)
[![Kedro](https://img.shields.io/badge/Kedro-1.0.0-8A2BE2.svg)](https://kedro.org/)
[![DVC](https://img.shields.io/badge/DVC-enabled-brightgreen.svg)](https://dvc.org/)
[![Airflow](https://img.shields.io/badge/Airflow-2.9.2-017CEE.svg)](https://airflow.apache.org/)
[![Docker](https://img.shields.io/badge/Docker-ready-0db7ed.svg)](https://www.docker.com/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

---

Este proyecto implementa un flujo completo de **Machine Learning** basado en la metodología **CRISP-DM**, gestionado con **Kedro**, versionado con **DVC**, y orquestado mediante **Apache Airflow** dentro de un entorno **Docker**.  

Se aplican modelos de **regresión y clasificación** sobre datos cinematográficos (`movies_metadata.csv`, `credits.csv`, `ratings.csv`) para entrenar, evaluar y versionar pipelines de ML.

---


```bash
## 🧩 Estructura del Proyecto

proyect-machine/
│
├── data/
│ ├── 01_raw/ # Datos originales
│ ├── 02_intermediate/ # Limpieza / feature engineering
│ ├── 03_primary/ # Datos preparados para modelos
│ └── 06_models/ # Modelos finales entrenados
│
├── src/
│ ├── proyect_machine/
│ │ ├── pipelines/
│ │ │ ├── modelo_regresion/
│ │ │ └── modelo_clasificacion/
│ │ ├── nodes/ # Funciones de entrenamiento y evaluación
│ │ └── hooks.py
│ └── ...
│
├── notebooks/ # Exploración, pruebas y resultados
├── conf/ # Configuración Kedro (catalog.yml, logging.yml)
├── models/
│ ├── regression/
│ └── classification/
│
├── docker-compose.yml # Contenedores Airflow + dependencias
├── requirements.txt
├── pyproject.toml
└── README.md

---

## ⚙️ Instalación y Configuración

1️⃣ Clonar el repositorio
git clone https://github.com/Arielv2004/Eva_2.machine_learning.git
cd proyect-machine

2️⃣ Crear entorno virtual (opcional en local)
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt


3️⃣ Construir el entorno Docker + Airflow + Kedro
docker compose up -d --build

Esto instala automáticamente:
Kedro 1.0.0
Apache Airflow 2.9.2
Pandas, NumPy, Scikit-learn
Kedro-Datasets y DVC

🚀 Ejecución de Pipelines (Airflow)

1. Ingresar a la interfaz web
Abre en tu navegador:
URL: http://localhost:8080
Usuario: admin
Contraseña: admin

2. DAG principal
run_kedro_pipelines → ejecuta ambos pipelines:
modelo_regresion
modelo_clasificacion

También puedes ejecutarlos manualmente dentro del contenedor:
docker exec -it proyect-machine-airflow-webserver bash
cd /opt/airflow/src/proyect-machine
kedro run --pipeline modelo_regresion
kedro run --pipeline modelo_clasificacion


🧠 Fases CRISP-DM
1. Business Understanding	Definición del objetivo: analizar factores que influyen en el éxito y calificación de películas.	
2. Data Understanding	Exploración de datasets (movies_metadata, credits, ratings), detección de nulos y correlaciones.	
3. Data Preparation	Limpieza, selección de variables y normalización.	
4. Modeling	Entrenamiento de modelos de regresión y clasificación.	
5. Evaluation	Métricas R², MAE, Accuracy, Precision, Recall, F1.
6. Deployment	Automatización con Airflow y versionado con DVC.	


🤖 Modelos Implementados

🔹 Regresión
Linear Regression (simple y múltiple)
Decision Tree Regressor
Random Forest Regressor
KNN Regressor

🔹 Clasificación
Logistic Regression
Decision Tree Classifier
KNN Classifier
Random Forest Classifier
SVM (RBF Kernel)

💾 Versionado con DVC
El proyecto utiliza DVC para rastrear los modelos entrenados y sus métricas.

dvc init
dvc add data/06_models/
git add .gitignore data/06_models.dvc
git commit -m "Track modelos con DVC"
🧩 Orquestación con Apache Airflow
El flujo completo (ETL → modelado → evaluación) se automatiza mediante DAGs ubicados en:
/opt/airflow/dags/run_kedro_pipelines.py

Los logs se almacenan automáticamente en:
C:/airflow_data/logs/


📈 Resultados
Regresión: el modelo Random Forest obtuvo el mejor R².
Clasificación: el modelo SVM (RBF) alcanzó la mayor precisión.
Los modelos se exportan como archivos .pkl en data/06_models/ y son versionados con DVC.

👨‍💻 Autores
Sergio Vera Sepúlveda
Ariel Velázquez

📚 Referencias
Kedro Documentation
Apache Airflow
DVC
CRISP-DM Methodology
