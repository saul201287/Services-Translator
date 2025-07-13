
import os
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import NearestNeighbors
import joblib

DATA_PATH = "data/dataset_zapoteco_basico.csv"
MODEL_DIR = "models"

def entrenar_y_guardar_modelos():
    print("📥 Cargando dataset...")
    df = pd.read_csv(DATA_PATH, encoding="utf-8")

    print("🧠 Entrenando modelo zapoteco → español...")
    vectorizer_zap = TfidfVectorizer(analyzer='char', ngram_range=(1, 3))
    X_zap = vectorizer_zap.fit_transform(df["zapoteco"])
    knn_zap = NearestNeighbors(n_neighbors=1, metric="cosine")
    knn_zap.fit(X_zap)

    # Modelo español → zapoteco
    print("🧠 Entrenando modelo español → zapoteco...")
    vectorizer_esp = TfidfVectorizer(analyzer='char', ngram_range=(1, 3))
    X_esp = vectorizer_esp.fit_transform(df["español"])
    knn_esp = NearestNeighbors(n_neighbors=1, metric="cosine")
    knn_esp.fit(X_esp)

    os.makedirs(MODEL_DIR, exist_ok=True)

    print("💾 Guardando modelos en carpeta 'modelos'...")
    joblib.dump(df, os.path.join(MODEL_DIR, "dataset.pkl"))
    joblib.dump(vectorizer_zap, os.path.join(MODEL_DIR, "vectorizer_zap.pkl"))
    joblib.dump(knn_zap, os.path.join(MODEL_DIR, "knn_zap.pkl"))
    joblib.dump(vectorizer_esp, os.path.join(MODEL_DIR, "vectorizer_esp.pkl"))
    joblib.dump(knn_esp, os.path.join(MODEL_DIR, "knn_esp.pkl"))

    print("✅ Modelos entrenados y guardados exitosamente.")


if __name__ == "__main__":
    print("🔁 INTERFAZ DE ENTRENAMIENTO DE MODELOS")
    print("1. Entrenar modelos desde cero")
    print("2. Reentrenar modelos con nuevos datos")
    opcion = input("Selecciona una opción (1 o 2): ").strip()

    if opcion in ("1", "2"):
        entrenar_y_guardar_modelos()
    else:
        print("❌ Opción no válida.")
