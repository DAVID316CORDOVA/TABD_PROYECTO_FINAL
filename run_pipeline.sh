#!/bin/bash

echo "===== Limpiando archivos JSON antiguos ====="
rm -f reseñas_restaurantes_api_final.json
rm -f restaurantes_bogota_multi_zona_final.json

echo "===== Iniciando scrapping en paralelo ====="

python scrapping_global.py &
PID_GLOBAL=$!

python scrapping_comentarios.py &
PID_COMENTARIOS=$!

echo "Esperando que terminen los scrapers..."
wait $PID_GLOBAL
wait $PID_COMENTARIOS

echo "===== Scrappers finalizados ====="

echo "===== Ejecutando postprocesamiento en paralelo ====="

python upload_data.py &
PID_UPLOAD=$!

python crear_indices_faiss.py &
PID_FAISS=$!

echo "Esperando que terminen upload_data y crear_indices_faiss..."
wait $PID_UPLOAD
wait $PID_FAISS

echo "===== Postprocesamiento completado ====="

echo "Lanzando Streamlit..."

streamlit run main.py
