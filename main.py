from fastapi import FastAPI, Query
import joblib
import os
from fuzzywuzzy import process

app = FastAPI(title="Traductor Zapoteco y tseltal ↔ Español")

# === Cargar modelos Zapoteco ===
ZAP_DIR = "models/zapoteco"
df_zap = joblib.load(os.path.join(ZAP_DIR, "dataset.pkl"))
vec_zap_zap = joblib.load(os.path.join(ZAP_DIR, "vectorizer_zap.pkl"))
knn_zap_zap = joblib.load(os.path.join(ZAP_DIR, "knn_zap.pkl"))
vec_zap_esp = joblib.load(os.path.join(ZAP_DIR, "vectorizer_esp.pkl"))
knn_zap_esp = joblib.load(os.path.join(ZAP_DIR, "knn_esp.pkl"))

# === Cargar modelos tseltal ===
TZEL_DIR = "models/tseltal"
df_tseltal = joblib.load(os.path.join(TZEL_DIR, "dataset.pkl"))
vec_tzel_zap = joblib.load(os.path.join(TZEL_DIR, "vectorizer_zap.pkl"))
knn_tzel_zap = joblib.load(os.path.join(TZEL_DIR, "knn_zap.pkl"))
vec_tzel_esp = joblib.load(os.path.join(TZEL_DIR, "vectorizer_esp.pkl"))
knn_tzel_esp = joblib.load(os.path.join(TZEL_DIR, "knn_esp.pkl"))

# === Funciones auxiliares ===
def fuzzy_cercana(input_str, lista, threshold=85, min_length=4):
    mejor, score = process.extractOne(input_str, lista)
    if score >= threshold and len(mejor) >= min_length:
        return mejor
    return None

def buscar_match(palabra, lista):
    palabra = palabra.strip().lower()
    if palabra in lista:
        return palabra
    return fuzzy_cercana(palabra, lista)

# === Endpoints raíz ===
@app.get("/")
def raiz():
    return {
        "mensaje": "API de traducción indígena ↔ español",
        "endpoints": [
            "/traducir/zapoteco", "/traducir/zapoteco-inverso",
            "/traducir/tseltal", "/traducir/tseltal-inverso"
        ]
    }

# === Zapoteco → Español ===
@app.get("/traducir/zapoteco")
def traducir_zapoteco(palabra: str):
    lista = df_zap["zapoteco"].dropna().tolist()
    match = buscar_match(palabra, lista)
    if match:
        vec = vec_zap_zap.transform([match])
        _, idx = knn_zap_zap.kneighbors(vec)
        traduccion = df_zap.iloc[idx[0][0]]["español"]
        return {"idioma": "zapoteco", "entrada": palabra, "match": match, "traduccion": traduccion}
    return {"error": "No se encontró traducción", "idioma": "zapoteco"}

# === Español → Zapoteco ===
@app.get("/traducir/zapoteco-inverso")
def traducir_zapoteco_inverso(palabra: str):
    lista = df_zap["español"].dropna().tolist()
    match = buscar_match(palabra, lista)
    if match:
        vec = vec_zap_esp.transform([match])
        _, idx = knn_zap_esp.kneighbors(vec)
        traduccion = df_zap.iloc[idx[0][0]]["zapoteco"]
        return {"idioma": "zapoteco", "entrada": palabra, "match": match, "traduccion": traduccion}
    return {"error": "No se encontró traducción", "idioma": "zapoteco"}

# === tseltal → Español ===
@app.get("/traducir/tseltal")
def traducir_tseltal(palabra: str):
    lista = df_tseltal["tseltal"].dropna().tolist()
    match = buscar_match(palabra, lista)
    if match:
        vec = vec_tzel_zap.transform([match])
        _, idx = knn_tzel_zap.kneighbors(vec)
        traduccion = df_tseltal.iloc[idx[0][0]]["español"]
        return {"idioma": "tseltal", "entrada": palabra, "match": match, "traduccion": traduccion}
    return {"error": "No se encontró traducción", "idioma": "tseltal"}

# === Español → tseltal ===
@app.get("/traducir/tseltal-inverso")
def traducir_tseltal_inverso(palabra: str):
    lista = df_tseltal["español"].dropna().tolist()
    match = buscar_match(palabra, lista)
    if match:
        vec = vec_tzel_esp.transform([match])
        _, idx = knn_tzel_esp.kneighbors(vec)
        traduccion = df_tseltal.iloc[idx[0][0]]["tseltal"]
        return {"idioma": "tseltal", "entrada": palabra, "match": match, "traduccion": traduccion}
    return {"error": "No se encontró traducción", "idioma": "tseltal"}
