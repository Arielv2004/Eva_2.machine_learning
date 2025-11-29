# Imagen base con Airflow ya instalado
FROM apache/airflow:2.9.2

# ============================
# 1) Dependencias del sistema
# ============================
USER root

RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        git \
        gcc \
        g++ \
        libxml2-dev \
        libxslt1-dev \
        libffi-dev \
        libssl-dev \
        libcurl4-openssl-dev \
        build-essential \
        python3-dev && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# Volvemos al usuario airflow (por seguridad)
USER airflow

# ============================
# 2) Carpeta de trabajo
# ============================
# En las imágenes de Airflow se usa /opt/airflow
WORKDIR /opt/airflow

# ============================
# 3) Dependencias de Python
# ============================
# Copiamos solo requirements.txt primero para aprovechar la cache
COPY requirements.txt ./requirements.txt

RUN pip install --no-cache-dir --upgrade pip setuptools wheel && \
    pip install --no-cache-dir -r requirements.txt && \
    pip install --no-cache-dir \
        kedro \
        "kedro-datasets[pandas]" \
        "dvc[s3]" \
        scikit-learn \
        pandas \
        numpy \
        matplotlib \
        seaborn \
        joblib \
        umap-learn \
        pyod \
        mlxtend \
        hdbscan \
        shap

# ============================
# 4) Copiar el proyecto
# ============================
COPY . .

# ============================
# 5) Telemetría de Kedro
# ============================
ENV KEDRO_DISABLE_TELEMETRY="yes"
