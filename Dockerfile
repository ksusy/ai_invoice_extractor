# syntax=docker/dockerfile:1
#
# Obraz s celou pipeline včetně systémových závislostí, které se instalují
# nejhůř — Tesseract s českým jazykovým balíkem a knihovny pro práci s obrazem.
#
#   docker build -t ai-invoice-extractor .
#   docker run --rm -p 8888:8888 ai-invoice-extractor            # JupyterLab
#   docker run --rm -p 8000:8000 ai-invoice-extractor api        # REST API
#
FROM python:3.11-slim-bookworm

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    MPLBACKEND=Agg

# tesseract-ocr-ces = český jazykový balík, bez něj OCR na fakturách nefunguje
RUN apt-get update && apt-get install -y --no-install-recommends \
        tesseract-ocr \
        tesseract-ocr-ces \
        libgl1 \
        libglib2.0-0 \
        poppler-utils \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Závislosti zvlášť, aby se vrstva s instalací neinvalidovala při každé změně kódu
COPY pyproject.toml README.md ./
COPY src/__init__.py ./src/
RUN pip install --upgrade pip && pip install -e ".[notebooks,pipeline,llm,dev]"

COPY . .

EXPOSE 8000 8888

COPY docker/entrypoint.sh /usr/local/bin/entrypoint.sh
# Odstranění CR: při sestavení na Windows by jinak shebang zněl "bash"
# a kontejner by skončil hláškou "/usr/bin/env: 'bash': No such file".
RUN sed -i 's/$//' /usr/local/bin/entrypoint.sh && chmod +x /usr/local/bin/entrypoint.sh

ENTRYPOINT ["entrypoint.sh"]
CMD ["notebooks"]
