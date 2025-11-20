import json
import numpy as np
import faiss
import openai
import os
from tqdm import tqdm

# Configuración
OPENAI_API_KEY = "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"
openai.api_key = OPENAI_API_KEY

JSON_FILE = "reseñas_restaurantes_api.json"  
OUTPUT_DIR = "./faiss_db"  
INDEX_FILE = os.path.join(OUTPUT_DIR, "resenas.index")
META_FILE = os.path.join(OUTPUT_DIR, "metadata.json")

print("=" * 70)
print("CREADOR DE ÍNDICE VECTORIAL FAISS PARA RESEÑAS")
print("=" * 70)

# Crear carpeta de salida si no existe
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Paso 1: Cargar reseñas desde JSON
print(f"\n[1/4] Cargando reseñas desde {JSON_FILE}...")

try:
    with open(JSON_FILE, "r", encoding="utf-8") as f:
        data = json.load(f)
    print(f" Archivo JSON cargado exitosamente")
    print(f" Restaurantes en el archivo: {len(data)}")
except FileNotFoundError:
    print(f" Error: No se encontró el archivo '{JSON_FILE}' en la carpeta actual")
    print(f"   Asegúrate de que el archivo esté en la misma carpeta que este script")
    exit()
except json.JSONDecodeError as e:
    print(f" Error al leer JSON: {e}")
    exit()

# Extraer todas las reseñas con su restaurante asociado
reseñas = []
for restaurante, reseñas_list in data.items():
    for reseña in reseñas_list:
        reseñas.append({
            "restaurante": restaurante,
            "texto": reseña["texto"],
            "fecha": reseña["fecha"]
        })

print(f" Total de reseñas extraídas: {len(reseñas)}")

if len(reseñas) == 0:
    print(" No se encontraron reseñas en el archivo JSON")
    exit()

# Paso 2: Generar embeddings con OpenAI
print(f"\n[2/4] Generando embeddings con OpenAI (modelo text-embedding-ada-002)...")
print(f"️  Este proceso puede tardar varios minutos dependiendo del número de reseñas")

embeddings_list = []
metadata_list = []

# Procesar en lotes de 100 para eficiencia
batch_size = 100
total_batches = (len(reseñas) + batch_size - 1) // batch_size

for i in tqdm(range(0, len(reseñas), batch_size), desc="Generando embeddings"):
    batch = reseñas[i:i + batch_size]
    textos = [item["texto"] for item in batch]

    try:
        # Llamar a OpenAI API
        response = openai.embeddings.create(
            model="text-embedding-ada-002",
            input=textos
        )

        # Extraer embeddings
        batch_embeddings = [d.embedding for d in response.data]
        embeddings_list.extend(batch_embeddings)

        # Guardar metadata
        for item in batch:
            metadata_list.append({
                "restaurante": item["restaurante"],
                "texto": item["texto"],
                "fecha": item["fecha"]
            })

    except Exception as e:
        print(f"\n Error en batch {i//batch_size + 1}/{total_batches}: {e}")
        continue

print(f"\n Embeddings generados: {len(embeddings_list)}")

if len(embeddings_list) == 0:
    print(" No se pudieron generar embeddings. Verifica tu API key de OpenAI")
    exit()

# Paso 3: Crear índice FAISS
print(f"\n[3/4] Creando índice FAISS...")

# Los embeddings de OpenAI text-embedding-ada-002 tienen 1536 dimensiones
dimension = 1536
embeddings_array = np.array(embeddings_list, dtype="float32")

# Crear índice FAISS (IndexFlatL2 para búsqueda exacta)
index = faiss.IndexFlatL2(dimension)

# Normalizar vectores para búsqueda por similitud coseno
faiss.normalize_L2(embeddings_array)

# Agregar vectores al índice
index.add(embeddings_array)

print(f" Índice FAISS creado con {index.ntotal} vectores")

# Paso 4: Guardar índice y metadata
print(f"\n[4/4] Guardando archivos...")

# Guardar índice FAISS
faiss.write_index(index, INDEX_FILE)
print(f" Índice guardado en: {INDEX_FILE}")

# Guardar metadata en JSON
with open(META_FILE, "w", encoding="utf-8") as f:
    json.dump(metadata_list, f, ensure_ascii=False, indent=2)
print(f" Metadata guardada en: {META_FILE}")

# Resumen final
print("\n" + "=" * 70)
print("ÍNDICE FAISS CREADO EXITOSAMENTE")
print("=" * 70)
print(f"\n Estadísticas:")
print(f"   - Restaurantes procesados: {len(data)}")
print(f"   - Total de reseñas: {len(reseñas)}")
print(f"   - Vectores en índice FAISS: {index.ntotal}")
print(f"   - Dimensiones por vector: {dimension}")
print(f"   - Tamaño archivo índice: {os.path.getsize(INDEX_FILE) / (1024 * 1024):.2f} MB")
print(f"   - Tamaño archivo metadata: {os.path.getsize(META_FILE) / (1024 * 1024):.2f} MB")

print(f"\n Archivos generados en carpeta '{OUTPUT_DIR}/':")
print(f"   - resenas.index (índice vectorial FAISS)")
print(f"   - metadata.json (información de reseñas)")

print("\n Proceso completado. Ahora puedes usar estos archivos en tu aplicación Streamlit.")
