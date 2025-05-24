FROM nvidia/cuda:12.9.0-cudnn-runtime-ubuntu24.04

# Install system dependencies (as root)
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

# Create output folder 
RUN mkdir -p /output 

# Set working directory
WORKDIR /app

# Copy everything as root
COPY . .
# Create metaflow folder and make it writable
RUN mkdir -p /app/.metaflow && chmod -R a+rwX /app/.metaflow

# Install Python dependencies
RUN pip install --no-cache-dir --break-system-packages -r requirements.txt

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV METAFLOW_DEFAULT_METADATA=local
ENV METAFLOW_DEFAULT_DATASTORE=local
ENV METAFLOW_USER=docker-user
ENV MODEL_DIR=/output

# Declare output volume
VOLUME ["/output"]

# Default command
CMD ["python", "training.py", "run"]
