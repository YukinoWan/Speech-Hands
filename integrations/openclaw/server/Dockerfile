FROM nvidia/cuda:12.1.1-cudnn8-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    PIP_NO_CACHE_DIR=1 \
    PYTHONUNBUFFERED=1

RUN apt-get update && apt-get install -y --no-install-recommends \
        python3.11 python3.11-venv python3-pip \
        ffmpeg git && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt /app/requirements.txt
RUN pip install --upgrade pip && pip install -r /app/requirements.txt

# The Speech-Hands research package must be installed on top of this
# image — either by mounting the repo and `pip install -e .` at runtime,
# or by baking a specific commit into a downstream image.
COPY app.py /app/app.py

ENV SPEECH_HANDS_CHECKPOINT=/models/speech-hands-qwen2.5-omni-7b \
    SPEECH_HANDS_DEVICE=cuda

EXPOSE 8080
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8080"]
