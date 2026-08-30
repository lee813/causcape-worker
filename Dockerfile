FROM pytorch/pytorch:2.6.0-cuda12.6-cudnn9-runtime

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1

# Install System Utilities & Media Toolchains
RUN apt-get update && apt-get install -y \
    git \
    ffmpeg \
    libsndfile1 \
    curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python Requirements
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Clone LongCat-Video and lock to verified commit
RUN git clone https://github.com/meituan-longcat/LongCat-Video.git /app/LongCat-Video && \
    cd /app/LongCat-Video && \
    git checkout 6b3f4b8582a8bc3f20f795735f5383716c4ba794

COPY handler.py .
COPY longcat_worker.py .

CMD ["python3", "-u", "handler.py"]
