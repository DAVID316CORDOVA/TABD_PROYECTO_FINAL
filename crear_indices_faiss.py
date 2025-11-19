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

# Configuración de APIs y Base de Datos


openai.api_key = OPENAI_API_KEY

DATABASE_NAME = "restaurantes_bogota_db"
COLLECTION_NAME = "bogota_data"

DB_PATH = "./faiss_db"
INDEX_FILE = os.path.join(DB_PATH, "resenas.index")
META_FILE = os.path.join(DB_PATH, "metadata.json")

# ======================
# FUNCIONES MONGODB CON ÍNDICES GEOESPACIALES
# ======================

def connect_mongo():
    """Establece conexión con MongoDB Atlas."""
    try:
        client = MongoClient(MONGO_URI, server_api=ServerApi('1'))
        return client[DATABASE_NAME][COLLECTION_NAME]
    except Exception as e:
        st.error(f"Error conectando a MongoDB: {e}")
        return None

def verificar_indice_geoespacial(col):
    """
    Verifica si existe el índice 2dsphere en MongoDB.
    Este índice permite búsquedas geoespaciales eficientes en O(log n).
    """
    try:
        indices = col.list_indexes()
        for idx in indices:
            if 'ubicacion_geo' in str(idx):
                return True
        return False
    except:
        return False

def buscar_restaurantes_cercanos_mongo(col, lat, lng, max_metros=3000):
    """
    Busca restaurantes usando el índice geoespacial 2dsphere de MongoDB.
    
    Proceso:
    1. Intenta usar índice geoespacial con query $nearSphere
       - MongoDB usa el índice idx_ubicacion_geo automáticamente
       - Retorna resultados ordenados por distancia en O(log n)
       - Mucho más rápido que recorrer todos los documentos
    
    2. Si no existe índice, hace búsqueda manual (fallback)
       - Lee todos los documentos O(n)
       - Calcula distancias manualmente con geopy
       - Filtra y ordena en memoria
    
    Parámetros:
        col: Colección de MongoDB
        lat, lng: Coordenadas del usuario
        max_metros: Radio de búsqueda en metros
    
    Retorna:
        DataFrame con restaurantes cercanos ordenados por distancia
    """
    try:
        if verificar_indice_geoespacial(col):
            # Usar índice geoespacial 2dsphere para búsqueda eficiente
            query = {
                "ubicacion_geo": {
                    "$nearSphere": {
                        "$geometry": {
                            "type": "Point",
                            "coordinates": [lng, lat]  # GeoJSON usa [lng, lat]
                        },
                        "$maxDistance": max_metros
                    }
                }
            }
            
            # MongoDB usa el índice y retorna ordenados por distancia
            resultados = list(col.find(query))
        else:
            # Fallback sin índice: búsqueda manual
            st.warning(" Índice geoespacial no encontrado. Usando búsqueda manual.")
            resultados = list(col.find({}))
        
        # Convertir ObjectId a string para pandas
        for doc in resultados:
            doc["_id"] = str(doc["_id"])
        
        df = pd.DataFrame(resultados)
        
        # Si no hay índice, filtrar manualmente por distancia
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

def calcular_distancia_manual(lat1, lng1, lat2, lng2):
    """Calcula distancia para mostrar al usuario (en metros)."""
    return geodesic((lat1, lng1), (lat2, lng2)).meters

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

def extract_coordinates(row):
    """Extrae coordenadas de formato GeoJSON o dict."""
    try:
        if 'ubicacion_geo' in row and isinstance(row['ubicacion_geo'], dict):
            coords = row['ubicacion_geo'].get('coordinates', [])
            if len(coords) == 2:
                return coords[1], coords[0]  # [lng, lat] -> (lat, lng)
        
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
    """Genera embeddings con OpenAI."""
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
    Normaliza el nombre del restaurante para hacer búsquedas flexibles.
    Ejemplo: "El Corral Cra 7 Calle 42" -> "el corral"
    """
    import re
    # Convertir a minúsculas
    nombre = nombre.lower().strip()
    # Remover direcciones comunes
    nombre = re.sub(r'\b(cra|carrera|calle|cl|kr|av|avenida|diagonal|dg|transversal|tv)\b.*', '', nombre)
    # Remover números de dirección
    nombre = re.sub(r'#\d+.*', '', nombre)
    nombre = re.sub(r'\d+-\d+', '', nombre)
    # Remover caracteres especiales y espacios extras
    nombre = re.sub(r'[^\w\s]', '', nombre)
    nombre = ' '.join(nombre.split())
    return nombre.strip()

def buscar_resenas_por_restaurante(index, metadata, nombre_restaurante: str, n_resultados=5):
    """
    Busca reseñas de un restaurante usando nombre normalizado.
    Maneja casos donde MongoDB tiene "El Corral Cra 7" y FAISS solo "El Corral"
    """
    nombre_normalizado = normalizar_nombre_restaurante(nombre_restaurante)
    
    # Filtrar metadata por restaurante con nombre normalizado
    resenas_restaurante = []
    for i, meta in enumerate(metadata):
        nombre_meta_normalizado = normalizar_nombre_restaurante(meta.get("restaurante", ""))
        # Buscar coincidencia parcial
        if nombre_normalizado in nombre_meta_normalizado or nombre_meta_normalizado in nombre_normalizado:
            resenas_restaurante.append((i, meta))
    
    if not resenas_restaurante:
        return []
    
    # Retornar las primeras n_resultados reseñas
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
    Genera resumen usando GPT solo si hay suficientes restaurantes con reseñas.
    No genera placeholders en corchetes, solo incluye restaurantes con datos reales.
    """
    texto_resenas = ""
    restaurantes_con_resenas = []
    
    # Recolectar restaurantes que tienen reseñas
    for r in restaurantes[:15]:
        nombre = get_restaurant_name(r)
        
        # Calcular distancia
        rest_lat, rest_lng = extract_coordinates(r)
        if rest_lat and rest_lng:
            distancia = calcular_distancia_manual(user_lat, user_lng, rest_lat, rest_lng)
        else:
            distancia = 0
        
        # Buscar reseñas con nombre normalizado
        resenas = buscar_resenas_por_restaurante(index, metadata, nombre, n_resultados=2)
        
        if resenas:
            restaurantes_con_resenas.append({
                'nombre': normalizar_nombre_restaurante(nombre).title(),
                'distancia': round(distancia),
                'rating': get_restaurant_rating(r),
                'resena': resenas[0]['texto'][:250]
            })
    
    # Solo generar resumen si hay al menos 3 restaurantes con reseñas
    if len(restaurantes_con_resenas) < 3:
        return None
    
    # Construir texto solo con restaurantes que tienen reseñas reales
    for rest in restaurantes_con_resenas[:5]:
        texto_resenas += f"- {rest['nombre']} (distancia {rest['distancia']}m, rating {rest['rating']})\n"
        texto_resenas += f"  Reseña: {rest['resena']}...\n\n"

    prompt = f"""
Analiza estos {len(restaurantes_con_resenas)} restaurantes cercanos y proporciona un resumen profesional:

{texto_resenas}

Genera un resumen que incluya:
1. Tipos de comida disponibles en la zona
2. Los 3 mejores restaurantes según las reseñas (SOLO los que tienen datos, NO inventes)
3. Aspectos destacados mencionados por los clientes

IMPORTANTE: Solo menciona restaurantes que aparecen en la lista anterior. No uses placeholders como [Restaurante 2] o [Comentario destacado]. Si solo hay información de 1-2 restaurantes, menciona solo esos.

Máximo 250 palabras, tono informativo.
"""
    
    try:
        completion = openai.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=400,
            temperature=0.7
        )
        return completion.choices[0].message.content
    
    except Exception as e:
        return None

# ======================
# INTERFAZ STREAMLIT
# ======================

st.set_page_config(page_title="Buscador de Restaurantes", layout="wide")
st.title(" Buscador Inteligente de Restaurantes")

# Conectar a MongoDB
col = connect_mongo()
if col is None:
    st.error("No se pudo conectar a la base de datos")
    st.stop()

# Cargar FAISS
index, metadata = cargar_faiss()
if index is None:
    st.stop()

# Cargar sistema silenciosamente
pass

# Input de usuario
addr = st.text_input(" Ingresa tu ubicación:", placeholder="Ej: Calle 72 Carrera 5, Bogotá")

# Slider para radio de búsqueda
radio_busqueda = st.slider("Radio de búsqueda (km)", min_value=0.5, max_value=5.0, value=3.0, step=0.5)

if addr:
    with st.spinner("🔍 Buscando tu ubicación..."):
        user_lat, user_lng = get_coordinates(addr)
    
    if user_lat and user_lng:
        # st.success(f" Ubicación encontrada: ({user_lat:.6f}, {user_lng:.6f})")
        pass
        
        with st.spinner(f"🔎 Buscando restaurantes en {radio_busqueda} km usando índices geoespaciales..."):
            # Usar índice 2dsphere de MongoDB
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
            
            # st.success(f" Se encontraron {len(nearby)} restaurantes en {radio_busqueda} km")

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

            # Generar resumen con IA solo si hay suficientes restaurantes con reseñas
            with st.spinner("Analizando restaurantes cercanos..."):
                summary = generar_resumen_con_ia(
                    nearby.to_dict(orient="records"), 
                    index, 
                    metadata,
                    user_lat,
                    user_lng
                )
            
            # Mostrar resumen solo si se generó correctamente
            if summary:
                st.subheader("🤖 Resumen Inteligente")
                st.write(summary)

            # Tabla de restaurantes
            st.subheader("📋 Lista de Restaurantes Cercanos")
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