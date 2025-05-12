#FROM python:3.10.8-slim
FROM nvidia/cuda:11.8.0-cudnn8-runtime-ubuntu22.04

# Set working directory to the project root
WORKDIR /Senior-AI-Engineer-Test

# Install system dependencies
# Install Python 3.10.8 and system dependencies
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    software-properties-common \
    && add-apt-repository ppa:deadsnakes/ppa \
    && apt-get update \
    && apt-get install -y --no-install-recommends \
    python3.10 \
    python3.10-dev \
    python3.10-distutils \
    python3.10-venv \
    wget \
    gcc \
    # OpenCV dependencies
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxrender1 \
    libxext6 \
    && rm -rf /var/lib/apt/lists/*

# Create a symbolic link for python
RUN ln -sf /usr/bin/python3.10 /usr/bin/python
# Install pip for Python 3.10
RUN wget https://bootstrap.pypa.io/get-pip.py && \
    python get-pip.py && \
    rm get-pip.py
# Copy requirements file from the learn directory
COPY learn/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Set PYTHONPATH to include your project
ENV PYTHONPATH=Senior-AI-Engineer-Test/learn

# Container will run as a non-root user for better security
RUN useradd -m labtech
USER labtech

# Default command (this will be overridden by the run script)
CMD ["python", "-c", "import sys; print(f'Python {sys.version} is ready to use!')"]
