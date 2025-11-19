# ===========================================
# VERIFICADOR DE ÍNDICES EN MONGODB
# ===========================================

from pymongo.mongo_client import MongoClient
from pymongo.server_api import ServerApi
import json
from pprint import pprint

# --- CONFIGURACIÓN ---
MONGO_URI = "mongodb+srv://topicos_user:vt2GV4Q75YFJrVpR@puj-topicos-bd.m302xsg.mongodb.net/?retryWrites=true&w=majority&appName=puj-topicos-bd"
DATABASE_NAME = "restaurantes_bogota_db"
COLLECTION_NAME = "bogota_data"

print("=" * 70)
print(" VERIFICADOR DE ÍNDICES EN MONGODB")
print("=" * 70)

try:
    # Conexión
    fixed_uri = MONGO_URI.replace('mongodb-srv://', 'mongodb+srv://')
    client = MongoClient(fixed_uri, server_api=ServerApi('1'))
    db = client[DATABASE_NAME]
    collection = db[COLLECTION_NAME]
    
    print(f"\n Base de datos: {DATABASE_NAME}")
    print(f" Colección: {COLLECTION_NAME}")
    
    # Contar el número de documentos en la colección
    total_restaurantes = collection.count_documents({})
    print(f"Total de restaurantes en la colección: {total_restaurantes}")
    
    
    # --- VERIFICAR ÍNDICES ---
    print("\n" + "=" * 70)
    print(" ÍNDICES ACTUALES:")
    print("=" * 70)
    
    indices = collection.index_information()
    
    if not indices or len(indices) == 1:  # Solo _id
        print("  No hay índices personalizados (solo _id por defecto)")
    else:
        for nombre_indice, info_indice in indices.items():
            print(f"\n Índice: {nombre_indice}")
            print(f"   Campos: {info_indice.get('key', [])}")
            
            # Identificar tipo de índice
            if '2dsphere' in str(info_indice.get('key', [])):
                print(f"   Tipo:  GEOESPACIAL (2dsphere)")
                print(f"    Optimizado para búsquedas geográficas ($near, $geoWithin)")
            elif 'text' in str(info_indice.get('key', [])):
                print(f"   Tipo:  TEXTO (text)")
                print(f"    Optimizado para búsquedas de texto completo")
            else:
                print(f"   Tipo:  ESTÁNDAR (B-tree)")
                print(f"    Optimizado para búsquedas de igualdad y rangos")
            
            if info_indice.get('unique'):
                print(f"    Restricción: UNIQUE (no permite duplicados)")
            if info_indice.get('sparse'):
                print(f"    SPARSE (solo indexa docs que tienen el campo)")
    
    # --- ESTADÍSTICAS DE LA COLECCIÓN ---
    print("\n" + "=" * 70)
    print(" ESTADÍSTICAS DE LA COLECCIÓN:")
    print("=" * 70)
    
    stats = db.command("collStats", COLLECTION_NAME)
    print(f" Total de documentos: {stats.get('count', 0):,}")
    print(f" Tamaño de datos: {stats.get('size', 0) / 1024:.2f} KB")
    print(f" Número de índices: {stats.get('nindexes', 0)}")
    print(f" Tamaño de índices: {stats.get('totalIndexSize', 0) / 1024:.2f} KB")
    
    # --- MUESTRA DE UN DOCUMENTO ---
    print("\n" + "=" * 70)
    print(" MUESTRA DE UN DOCUMENTO (para ver estructura):")
    print("=" * 70)
    
    sample = collection.find_one()
    if sample:
        # Ocultar _id para mejor legibilidad
        sample.pop('_id', None)
        print(json.dumps(sample, indent=2, ensure_ascii=False))
    else:
        print("  La colección está vacía")
    
    print("\n" + "=" * 70)
    print(" VERIFICACIÓN COMPLETADA")
    print("=" * 70)

except Exception as e:
    print(f"\n ERROR: {e}")

finally:
    if 'client' in locals():
        client.close()
        print("\n🔌 Conexión cerrada")