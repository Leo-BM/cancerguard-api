# ─── Fase 8: CancerGuard API — Imagem da FastAPI ─────────────────────────────
#
# Estratégia de layers para maximizar cache do Docker:
#   1. Base (nunca muda)
#   2. requirements.txt → pip install (lento ~2min, cache reutilizado)
#   3. Código da app (rápido, invalida só quando há mudança de código)
#
FROM python:3.13-slim

WORKDIR /app

# shap==0.46.0 precisa compilar extensão C++ (_cext.cc) no Linux.
# Instalamos g++ antes do pip, e removemos após para não inflar a imagem.
RUN apt-get update && apt-get install -y --no-install-recommends g++ \
    && rm -rf /var/lib/apt/lists/*

# Layer — dependências (cache enquanto requirements.txt não mudar)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Layer 3 — código da aplicação
COPY app/ ./app/
COPY training/ ./training/

EXPOSE 8000

# --host 0.0.0.0 é obrigatório em containers:
# sem ele, uvicorn escuta apenas em localhost do container
# e o host externo não consegue se conectar via porta mapeada.
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
