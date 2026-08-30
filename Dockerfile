FROM nvidia/cuda:12.4.1-cudnn-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1

# Install System Utilities, Audio/Video Libs, and Build Tools
RUN apt-get update && apt-get install -y \
    python3.10 \
    python3-pip \
    python3-dev \
    git \
    ffmpeg \
    libsndfile1 \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

RUN python3 -m pip install --upgrade pip setuptools wheel

WORKDIR /app

# Install Python Dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Clone Official LongCat-Video repository and lock to verified commit
RUN git clone https://github.com/meituan-longcat/LongCat-Video.git /app/LongCat-Video && \
    cd /app/LongCat-Video && \
    git checkout 6b3f4b8582a8bc3f20f795735f5383716c4ba794

# Copy Worker Code and Handler
COPY handler.py .
COPY longcat_worker.py .

# RunPod Serverless Entrypoint
CMD ["python3", "-u", "handler.py"]
