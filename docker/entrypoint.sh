#!/usr/bin/env bash
# Volba režimu kontejneru. Bez argumentu se spustí JupyterLab s notebooky.
set -e

case "${1:-notebooks}" in
  notebooks)
    echo "JupyterLab poběží na http://localhost:8888  (bez hesla a tokenu)"
    exec jupyter lab \
      --ip=0.0.0.0 --port=8888 --no-browser --allow-root \
      --ServerApp.token='' --ServerApp.password='' \
      --notebook-dir=/app/experiments/notebooks
    ;;
  api)
    echo "REST API poběží na http://localhost:8000/docs"
    exec uvicorn src.main:app --host 0.0.0.0 --port 8000
    ;;
  test)
    exec pytest
    ;;
  shell)
    shift
    exec /bin/bash "$@"
    ;;
  *)
    exec "$@"
    ;;
esac
