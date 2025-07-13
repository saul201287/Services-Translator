from fastapi import FastAPI, Query
import joblib
import os
from fuzzywuzzy import process

app = FastAPI(title="Traductor Zapoteco-Español (versión sin category)")

MODEL_DIR = "models"
df = joblib.load(os.path.join(MODEL_DIR, "dataset.pkl"))
vectorizer_zap = joblib.load(os.path.join(MODEL_DIR, "vectorizer_zap.pkl"))
knn_zap = joblib.load(os.path.join(MODEL_DIR, "knn_zap.pkl"))
vectorizer_esp = joblib.load(os.path.join(MODEL_DIR, "vectorizer_esp.pkl"))
knn_esp = joblib.load(os.path.join(MODEL_DIR, "knn_esp.pkl"))

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

@app.get("/")
def raiz():
    return {"mensaje": "API de traducción Zapoteco ↔ Español (sin columna category)"}

@app.get("/traducir")
def traducir(palabra: str = Query(..., description="Palabra o frase en zapoteco")):
    zapoteco_lista = df["zapoteco"].dropna().tolist()
    match = buscar_match(palabra, zapoteco_lista)
    if match:
        vec = vectorizer_zap.transform([match])
        dist, idx = knn_zap.kneighbors(vec)
        traduccion = df.iloc[idx[0][0]]["español"]
        return {"entrada": palabra, "match": match, "traduccion": traduccion}
    return {
        "error": "No se encontró una traducción similar.",
        "sugerencia": "Verifica la ortografía o prueba con otra palabra/frase conocida."
    }

@app.get("/traducir-inverso")
def traducir_inverso(palabra: str = Query(..., description="Palabra o frase en español")):
    espanol_lista = df["español"].dropna().tolist()
    match = buscar_match(palabra, espanol_lista)
    if match:
        vec = vectorizer_esp.transform([match])
        dist, idx = knn_esp.kneighbors(vec)
        traduccion = df.iloc[idx[0][0]]["zapoteco"]
        return {"entrada": palabra, "match": match, "traduccion": traduccion}
    return {
        "error": "No se encontró una traducción similar.",
        "sugerencia": "Verifica la ortografía o prueba con otra palabra/frase conocida."
    }
