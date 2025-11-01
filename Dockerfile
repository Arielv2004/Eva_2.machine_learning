# ============================================
# 🧠 Dockerfile - Proyecto ML con Kedro + DVC
# ============================================

# Imagen base ligera pero compatible
FROM python:3.11-slim

# Carpeta de trabajo
WORKDIR /app/proyect-machine

# Copiar proyecto
COPY proyect-machine /app/proyect-machine
COPY proyect-machine/requirements.txt /app/requirements.txt
COPY .dvc /app/.dvc
COPY .dvcignore /app/.dvcignore

# Instalar dependencias del sistema necesarias para compilar y DVC/lxml
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

# Instalar dependencias Python
RUN pip install --no-cache-dir --upgrade pip setuptools wheel && \
    pip install --no-cache-dir -r /app/requirements.txt && \
    pip install --no-cache-dir \
        kedro \
        kedro-datasets[pandas] \
        dvc[s3] \
        scikit-learn \
        pandas \
        numpy \
        matplotlib \
        seaborn \
        joblib

# Desactivar telemetría de Kedro
ENV KEDRO_DISABLE_TELEMETRY="yes"

# Comando por defecto
CMD ["kedro", "run"]
