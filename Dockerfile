FROM python:3.11-slim

RUN apt-get update \
    && apt-get install -y --no-install-recommends ffmpeg libsndfile1 \
    && rm -rf /var/lib/apt/lists/*

ENV PYTHONUNBUFFERED=1 \
    TORCH_HOME=/models \
    OMP_NUM_THREADS=4

WORKDIR /app

# PyTorch version CPU (beaucoup plus légère que la version GPU)
RUN pip install --no-cache-dir torch==2.5.1 torchaudio==2.5.1 \
    --index-url https://download.pytorch.org/whl/cpu

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Télécharge le modèle au build pour éviter de le re-télécharger à chaque démarrage
RUN python -c "from demucs.pretrained import get_model; get_model('htdemucs')"

COPY bot.py .

CMD ["python", "bot.py"]
