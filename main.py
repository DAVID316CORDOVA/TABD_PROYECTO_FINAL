import os
import json
import pandas as pd
import requests
from pymongo.mongo_client import MongoClient
from pymongo.server_api import ServerApi
from geopy.distance import geodesic
import streamlit as st
import folium
from streamlit_folium import st_folium
import openai
import numpy as np
import faiss
from typing import List, Dict
import re

# Configuración de APIs y Base de Datos

openai.api_key = OPENAI_API_KEY

OPENAI_API_KEY = st.secrets["OPENAI_API_KEY"]
GOOGLE_KEY = st.secrets["GOOGLE_KEY"]
MONGO_URI = st.secrets["MONGO_URI"]

DATABASE_NAME = "restaurantes_bogota_db"
COLLECTION_NAME = "bogota_data"

DB_PATH = "./faiss_db"
INDEX_FILE = os.path.join(DB_PATH, "resenas.index")
META_FILE = os.path.join(DB_PATH, "metadata.json")

# ======================
# FUNCIONES MONGODB
# ======================

def connect_mongo():
    """Conecta a MongoDB Atlas y retorna la colección."""
    try:
        client = MongoClient(MONGO_URI, server_api=ServerApi('1'))
        return client[DATABASE_NAME][COLLECTION_NAME]
    except Exception as e:
        st.error(f"Error conectando a MongoDB: {e}")
        return None

def verificar_indice_geoespacial(col):
    """Verifica si existe el índice 2dsphere en la colección."""
    try:
        indices = col.list_indexes()
        for idx in indices:
            if 'ubicacion_geo' in str(idx):
                return True
        return False
    except:
        return False

def buscar_restaurantes_cercanos_mongo(col, lat, lng, max_metros=4000):
    """
    Busca restaurantes cercanos usando índice geoespacial de MongoDB.
    
    Si existe índice 2dsphere:
        - Usa query $nearSphere para búsqueda eficiente O(log n)
        - MongoDB retorna resultados ordenados por distancia automáticamente
    
    Si no existe índice:
        - Lee todos los documentos O(n)
        - Calcula distancias manualmente
        - Filtra y ordena en memoria
    """
    try:
        if verificar_indice_geoespacial(col):
            # Usar índice geoespacial
            query = {
                "ubicacion_geo": {
                    "$nearSphere": {
                        "$geometry": {
                            "type": "Point",
                            "coordinates": [lng, lat]
                        },
                        "$maxDistance": max_metros
                    }
                }
            }
            resultados = list(col.find(query))
        else:
            # Fallback sin índice
            resultados = list(col.find({}))
        
        for doc in resultados:
            doc["_id"] = str(doc["_id"])
        
        df = pd.DataFrame(resultados)
        
        # Si no hay índice, filtrar manualmente
        if not verificar_indice_geoespacial(col) and not df.empty:
            distancias = []
            indices_validos = []
            
            for idx, row in df.iterrows():
                rest_lat, rest_lng = extract_coordinates(row)
                if rest_lat and rest_lng:
                    dist = calcular_distancia_manual(lat, lng, rest_lat, rest_lng)
                    if dist <= max_metros:
                        distancias.append(dist)
                        indices_validos.append(idx)
            
            if indices_validos:
                df = df.loc[indices_validos].copy()
                df['dist_temp'] = distancias
                df = df.sort_values('dist_temp').drop('dist_temp', axis=1)
        
        return df
    
    except Exception as e:
        st.error(f"Error en búsqueda: {e}")
        return pd.DataFrame()

# ======================
# FUNCIONES AUXILIARES
# ======================

def get_coordinates(address):
    """Geocodifica dirección usando Google Maps API."""
    base_url = "https://maps.googleapis.com/maps/api/geocode/json"
    r = requests.get(base_url, params={"address": address + ", Bogotá, Colombia", "key": GOOGLE_KEY}).json()
    if r["status"] == "OK":
        loc = r["results"][0]["geometry"]["location"]
        return loc["lat"], loc["lng"]
    return None, None

def calcular_distancia_manual(lat1, lng1, lat2, lng2):
    """Calcula distancia geodésica en metros."""
    return geodesic((lat1, lng1), (lat2, lng2)).meters

def extract_coordinates(row):
    """Extrae coordenadas de formato GeoJSON o dict."""
    try:
        if 'ubicacion_geo' in row and isinstance(row['ubicacion_geo'], dict):
            coords = row['ubicacion_geo'].get('coordinates', [])
            if len(coords) == 2:
                return coords[1], coords[0]
        
        if 'ubicacion' in row and isinstance(row['ubicacion'], dict):
            lat = row['ubicacion'].get('lat')
            lng = row['ubicacion'].get('lng')
            if lat is not None and lng is not None:
                return float(lat), float(lng)
        
        return None, None
    except:
        return None, None

def get_restaurant_name(r):
    """Extrae nombre del restaurante."""
    return r.get('nombre') or r.get('Nombre') or r.get('name') or 'Sin nombre'

def get_restaurant_rating(r):
    """Extrae rating del restaurante."""
    return r.get('rating') or r.get('Rating') or 'N/A'

# ======================
# FUNCIONES FAISS
# ======================

def obtener_embeddings_openai(textos: List[str]) -> np.ndarray:
    """Genera embeddings vectoriales con OpenAI."""
    response = openai.embeddings.create(model="text-embedding-ada-002", input=textos)
    return np.array([d.embedding for d in response.data], dtype="float32")

def cargar_faiss():
    """Carga índice FAISS y metadata desde disco."""
    if not os.path.exists(INDEX_FILE) or not os.path.exists(META_FILE):
        st.error("No se encontró la base FAISS. Ejecuta crear_indices_faiss.py primero")
        return None, None
    
    index = faiss.read_index(INDEX_FILE)
    
    with open(META_FILE, "r", encoding="utf-8") as f:
        metadata = json.load(f)
    
    return index, metadata

def normalizar_nombre_restaurante(nombre: str) -> str:
    """
    Normaliza nombres para coincidencias flexibles entre MongoDB y FAISS.
    Ejemplo: "El Corral Cra 7 Calle 42" -> "el corral"
    """
    nombre = nombre.lower().strip()
    nombre = re.sub(r'\b(cra|carrera|calle|cl|kr|av|avenida|diagonal|dg|transversal|tv)\b.*', '', nombre)
    nombre = re.sub(r'#\d+.*', '', nombre)
    nombre = re.sub(r'\d+-\d+', '', nombre)
    nombre = re.sub(r'[^\w\s]', '', nombre)
    nombre = ' '.join(nombre.split())
    return nombre.strip()

def buscar_resenas_por_restaurante(index, metadata, nombre_restaurante: str, n_resultados=5):
    """
    Busca reseñas en FAISS usando coincidencia flexible de nombres.
    Permite que "El Corral Cra 7" (MongoDB) coincida con "El Corral" (FAISS).
    """
    nombre_normalizado = normalizar_nombre_restaurante(nombre_restaurante)
    
    resenas_restaurante = []
    for i, meta in enumerate(metadata):
        nombre_meta_normalizado = normalizar_nombre_restaurante(meta.get("restaurante", ""))
        if nombre_normalizado in nombre_meta_normalizado or nombre_meta_normalizado in nombre_normalizado:
            resenas_restaurante.append((i, meta))
    
    if not resenas_restaurante:
        return []
    
    resultados = []
    for _, meta in resenas_restaurante[:n_resultados]:
        resultados.append({
            "texto": meta.get("texto", ""),
            "restaurante": meta.get("restaurante", ""),
            "fecha": meta.get("fecha", ""),
            "score": 1.0
        })
    
    return resultados

def generar_resumen_con_ia(restaurantes: List[Dict], index, metadata, user_lat, user_lng):
    """
    Genera resumen de los 5 restaurantes MÁS CERCANOS que tienen reseñas en FAISS.
    
    Proceso:
    1. Ordena restaurantes por distancia (ya vienen ordenados de MongoDB)
    2. Busca reseñas en FAISS para cada uno
    3. Toma los 5 primeros que SÍ tienen reseñas
    4. Genera análisis con GPT usando esas reseñas
    """
    restaurantes_con_resenas = []
    
    # Recorrer restaurantes ordenados por distancia y buscar reseñas en FAISS
    for r in restaurantes:
        nombre = get_restaurant_name(r)
        
        rest_lat, rest_lng = extract_coordinates(r)
        if rest_lat and rest_lng:
            distancia = calcular_distancia_manual(user_lat, user_lng, rest_lat, rest_lng)
        else:
            continue
        
        # Buscar reseñas en la base vectorial FAISS
        resenas = buscar_resenas_por_restaurante(index, metadata, nombre, n_resultados=3)
        
        # Solo agregar si tiene reseñas
        if resenas:
            restaurantes_con_resenas.append({
                'nombre': normalizar_nombre_restaurante(nombre).title(),
                'nombre_original': nombre,
                'distancia': round(distancia),
                'rating': get_restaurant_rating(r),
                'resenas': resenas
            })
            
            # Detener cuando tengamos los 5 más cercanos con reseñas
            if len(restaurantes_con_resenas) >= 5:
                break
    
    # Si no se encontraron reseñas
    if len(restaurantes_con_resenas) == 0:
        return "No se encontraron reseñas en la base vectorial FAISS para ninguno de los restaurantes cercanos."
    
    # Construir texto con los 5 restaurantes más cercanos que tienen reseñas
    texto_resenas = ""
    for i, rest in enumerate(restaurantes_con_resenas, 1):
        texto_resenas += f"\n{i}. {rest['nombre']} ({rest['distancia']}m, rating {rest['rating']})\n"
        for j, resena in enumerate(rest['resenas'][:2], 1):
            texto_resenas += f"   Reseña {j}: {resena['texto'][:250]}...\n"

    prompt = f"""Analiza estos {len(restaurantes_con_resenas)} restaurantes MÁS CERCANOS basándote en las reseñas reales de clientes:

{texto_resenas}

Genera un análisis completo que incluya:

1. **Resumen de cada restaurante**: Breve descripción de qué ofrece cada uno según las reseñas

2. **Recomendaciones**: Cuál elegir según diferentes necesidades:
   - Mejor calidad según clientes
   - Mejor relación calidad-precio
   - Mejor ambiente/servicio

3. **Aspectos destacados**: Lo que más valoran los clientes en estos lugares

IMPORTANTE:
- Estos son los restaurantes MÁS CERCANOS a la ubicación del usuario
- Solo menciona los restaurantes listados arriba
- Basa tu análisis en las reseñas reales proporcionadas
- NO inventes información

Máximo 350 palabras, tono profesional y útil."""
    
    try:
        completion = openai.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=600,
            temperature=0.7
        )
        
        # Agregar información de cuántos restaurantes se analizaron
        resumen_header = f"*Análisis basado en los {len(restaurantes_con_resenas)} restaurantes más cercanos *\n\n"
        
        return resumen_header + completion.choices[0].message.content
        
    except Exception as e:
        return f"Error al generar resumen con GPT: {str(e)}\n\nSe encontraron reseñas para {len(restaurantes_con_resenas)} restaurantes en FAISS."

# ======================
# INTERFAZ STREAMLIT
# ======================

st.set_page_config(page_title="Buscador de Restaurantes", layout="wide")
st.title(" Buscador Inteligente de Restaurantes")

col = connect_mongo()
if col is None:
    st.error("No se pudo conectar a la base de datos")
    st.stop()

index, metadata = cargar_faiss()
if index is None:
    st.stop()

# Contar total de restaurantes en MongoDB
total_restaurantes = col.count_documents({})
st.caption(f" Total de restaurantes en la base de datos: {total_restaurantes}")

addr = st.text_input(" Ingresa tu ubicación:", placeholder="Ej: Calle 72 Carrera 5, Bogotá")

radio_busqueda = st.slider("Radio de búsqueda (km)", min_value=0.5, max_value=10.0, value=4.0, step=0.5)

if addr:
    with st.spinner(" Buscando tu ubicación..."):
        user_lat, user_lng = get_coordinates(addr)
    
    if user_lat and user_lng:
        
        with st.spinner(f" Buscando restaurantes en {radio_busqueda} km..."):
            nearby = buscar_restaurantes_cercanos_mongo(col, user_lat, user_lng, max_metros=radio_busqueda * 1000)
        
        if nearby.empty:
            st.warning(f" No se encontraron restaurantes en un radio de {radio_busqueda} km")
        else:
            # Calcular distancias para mostrar
            distancias = []
            for idx, row in nearby.iterrows():
                rest_lat, rest_lng = extract_coordinates(row)
                if rest_lat and rest_lng:
                    dist = calcular_distancia_manual(user_lat, user_lng, rest_lat, rest_lng)
                    distancias.append(dist)
                else:
                    distancias.append(float('inf'))
            
            nearby["dist"] = distancias
            
            st.info(f" Se encontraron **{len(nearby)} restaurantes** en un radio de {radio_busqueda} km")

            # Mapa
            m = folium.Map(location=[user_lat, user_lng], zoom_start=14)
            
            folium.Marker(
                [user_lat, user_lng],
                tooltip="Tu ubicación",
                icon=folium.Icon(color="red", icon="home", prefix='fa')
            ).add_to(m)
            
            for idx, row in nearby.iterrows():
                lat, lng = extract_coordinates(row)
                if lat and lng:
                    folium.Marker(
                        [lat, lng],
                        tooltip=f"{get_restaurant_name(row)}\nRating: {get_restaurant_rating(row)}\nDistancia: {round(row.get('dist',0))}m",
                        popup=f"<b>{get_restaurant_name(row)}</b><br>Rating: {get_restaurant_rating(row)}<br>Distancia: {round(row.get('dist',0))}m",
                        icon=folium.Icon(color="blue", icon="cutlery", prefix='fa')
                    ).add_to(m)
            
            st.subheader(" Mapa de Restaurantes")
            st_folium(m, width=900, height=500)

            # Generar reseña inteligente de los 5 más cercanos con reseñas
            st.subheader(" Reseña Inteligente de los Restaurantes Más Cercanos")
            with st.spinner("Buscando las reseñas de los 5 restaurantes más cercanos..."):
                summary = generar_resumen_con_ia(
                    nearby.to_dict(orient="records"), 
                    index, 
                    metadata,
                    user_lat,
                    user_lng
                )
            
            st.write(summary)

            # Tabla de restaurantes
            st.subheader(" Lista de Restaurantes Cercanos")
            display_data = []
            for idx, row in nearby.iterrows():
                display_data.append({
                    'Nombre': get_restaurant_name(row),
                    'Rating': get_restaurant_rating(row),
                    'Distancia (m)': round(row.get('dist', 0)),
                    'Dirección': row.get('direccion') or row.get('Dirección') or row.get('address') or 'N/A'
                })
            st.dataframe(pd.DataFrame(display_data), use_container_width=True, hide_index=True)
    else:

        st.error(" No se pudo encontrar la ubicación. Intenta con una dirección más específica.")
