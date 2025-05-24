FROM nvidia/cuda:11.8.0-cudnn8-runtime-ubuntu22.04

# Install system dependencies
RUN apt-get update && apt-get install -y \
    python3 \
    python3-pip \
    build-essential \
    curl \
    unzip \
    && rm -rf /var/lib/apt/lists/*

# Create symlinks for python
RUN ln -sf /usr/bin/python3 /usr/bin/python && \
    ln -sf /usr/bin/pip3 /usr/bin/pip

# Set working directory
WORKDIR /app

# Copy and install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy all source code
COPY . .

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV METAFLOW_DEFAULT_METADATA=local
ENV METAFLOW_DEFAULT_DATASTORE=local
ENV METAFLOW_USER=docker-user
ENV MODEL_DIR=/output

# Create output folder for reports/artifacts
RUN mkdir -p /output && chmod 777 /output
VOLUME ["/output"]

# Set the default command for training
CMD ["python", "training.py", "run"]
