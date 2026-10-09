# Imagen para Hugging Face Spaces (SDK Docker) / Render / Railway.
FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1 HF_HOME=/home/user/.cache/huggingface PORT=7860

RUN useradd -m -u 1000 user
WORKDIR /app

# PyTorch solo-CPU (evita descargar ~2 GB de CUDA)
COPY requirements.txt .
RUN pip install torch --index-url https://download.pytorch.org/whl/cpu \
 && pip install -r requirements.txt

COPY --chown=user . .
USER user

# Pre-construye el índice y descarga el modelo de embeddings durante el build (no requiere API key)
RUN python -m scripts.build_index

EXPOSE 7860
CMD ["sh", "-c", "gunicorn app:app --bind 0.0.0.0:${PORT} --workers 1 --threads 4 --timeout 180"]
