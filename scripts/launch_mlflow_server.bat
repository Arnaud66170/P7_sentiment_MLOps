@echo off
echo 🚀 Démarrage du MLFlow Tracking Server...
mlflow server --backend-store-uri ./mlruns --default-artifact-root ./mlruns --host 127.0.0.1 --port 5000
pause