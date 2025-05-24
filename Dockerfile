FROM nvidia/cuda:12.9.0-cudnn-runtime-ubuntu24.04

# Create a non-root user
RUN useradd -ms /bin/bash appuser

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

# Create output folder and assign ownership
RUN mkdir -p /output && chown -R appuser:appuser /output

# Switch to non-root user (now that all root-level setup is done)
USER appuser

# Set working directory
WORKDIR /app

# Copy and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir --break-system-packages -r requirements.txt

# Copy all source code
COPY . .

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV METAFLOW_DEFAULT_METADATA=local
ENV METAFLOW_DEFAULT_DATASTORE=local
ENV METAFLOW_USER=appuser
ENV MODEL_DIR=/output

# Declare output volume
VOLUME ["/output"]

# Set the default command for training
CMD ["python", "training.py", "run"]
