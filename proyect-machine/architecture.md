1. Visión General del Proyecto

El proyecto implementa un flujo completo de Machine Learning siguiendo la metodología CRISP-DM, apoyado en:
Kedro para la creación de pipelines modulares.
DVC para versionamiento de datos y modelos.
Airflow para orquestación de los pipelines.
Docker para un entorno reproducible.
Jupyter Notebooks para análisis exploratorio y validación de modelos.

El sistema combina tareas de data engineering, modelos supervisados, modelos no supervisados y reportes finales para generar un pipeline end-to-end.

2. Estructura de Carpetas

La estructura principal del proyecto es:

proyect-machine/
│
├── conf/                         
├── dags/                         
├── Data/                         
├── models/ 
│   ├── classification.dvc/
│   ├── regression.dvc/     
├── dogs/                    
├── notebooks/    
│   ├── 04_modelos_clasificacion/ 
│   │     ├── 01_logistic_regression/
│   │     ├── 02_decicion_tree/
│   │     ├── 03_svc/
│   │     ├── 04_knn/
│   │     ├── 05_random_forest/ 
│   ├── 04_modelos_regresion/
│   │     ├── 01_lineal_simple/
│   │     ├── 02_lineal_multiple/
│   │     ├── 03_decision_tree_regressor/
│   │     ├── 04_random_forest/
│   │     ├── 05_regresion_knn/
│   ├── 01_business_understanding/ 
│   ├── 02_data_understanding/ 
│   ├── 03_data_preparation/ 
│   ├── 05_evaluation/ 
│   ├── 06_unsupervised_learning/ 
│   ├── 07_final_analysis/                
├── src/proyect_machine/          
│   ├── pipelines/
│   │   ├── modelo_regresion/
│   │   ├── modelo_clasificacion/
│   │   ├── movies_pipeline/
│   │   └── unsupervised_learning/
│   │        └── clustering/
│   │        └── dimensionality_reduction/
│   ├── pipeline_registry.py
│   ├── settings.py
│   └── __main__.py
├── tests/           

3. Arquitectura de Pipelines
3.1. movies_pipeline 

Este pipeline prepara los datos base utilizados por los modelos:

Nodo Función Descripción
clean_movies_metadata	Limpieza	Elimina nulos en columnas críticas como title y release_date.
merge_movies_credits	Unión	Combina metadata con credits.csv usando el ID de película.
process_ratings	Agregación	Calcula promedio de ratings por película.

3.2. Pipeline Supervisado — Regresión (modelo_regresion)

Este pipeline predice una variable continua 
Incluye:

división train/test
normalización
entrenamiento de modelos:
Linear Regression
Decision Tree Regressor
Random Forest
KNN
evaluación (MAE, RMSE, R²)
guardado y versionamiento del mejor modelo

3.3. Pipeline Supervisado — Clasificación (modelo_clasificacion)

Este pipeline clasifica observaciones 

Modelos incluidos:
Logistic Regression
KNN
Decision Tree
Random Forest
SVM
Métricas:
Accuracy
Precision
Recall
F1-score

3.4. Pipeline No Supervisado (unsupervised_learning)

Este pipeline implementa varias técnicas de clustering y reducción dimensional:

Clustering
K-Means
DBSCAN
Gaussian Mixture Models (GMM)

Incluye:
Silhouette Score
Davies-Bouldin
Calinski-Harabasz
Elbow Method
Dendrograma (hierarchical)
Reducción de dimensionalidad

PCA
t-SNE
Salidas
cluster_kmeans
cluster_dbscan
cluster_gmm
X_pca.parquet
X_tsne.parquet
Estas salidas se versionan en el directorio 07_model_output/.

4. Integración con DVC

Todos los artefactos importantes se versionan:
modelos entrenados (pickle)
salidas de clustering
representaciones PCA/t-SNE
métricas de cada experimento
Archivo generado:
data/06_models.dvc
data/07_model_output.dvc

5. Orquestación con Airflow

Airflow ejecuta automaticamente:
data_engineering → supervised_learning → unsupervised_learning

El DAG se encuentra en:
dags/run_kedro_pipelines.py


6. Arquitectura Docker

El docker-compose.yml levanta:
webserver Airflow
scheduler
flower 
worker
entorno Kedro dentro del contenedor
Esto garantiza que el profe pueda ejecutar todo el proyecto sin instalar dependencias.

7. Notebooks Incluidos

 modelos_clasificacion
    01_logistic_regression
    02_decicion_tree
    03_svc
    04_knn
    05_random_forest
modelos_regresion
    01_lineal_simple
    02_lineal_multiple
    03_decision_tree_regressor
04_random_forest
05_regresion_knn
01_business_understanding
02_data_understanding
03_data_preparation
05_evaluation
06_unsupervised_learning
07_final_analysis 


9. Conclusión

La arquitectura del proyecto permite:
reproducibilidad completa
separación clara entre etapas
experimentación controlada
escalabilidad mediante Airflow y Docker
pipelines consistentes gracias a Kedro
Este diseño cumple con los estándares profesionales para un proyecto de Machine Learning end-to-end