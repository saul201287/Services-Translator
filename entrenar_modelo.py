import os
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import NearestNeighbors
import joblib

IDIOMAS = {
    "zapoteco": {
        "csv": "data/dataset_zapoteco_basico.csv",
        "col_idioma": "zapoteco",
        "col_espanol": "español",
        "modelo_dir": "models/zapoteco"
    },
    "tzeltal": {
        "csv": "data/diccionario_tseltal.csv",
        "col_idioma": "tseltal",
        "col_espanol": "español",
        "modelo_dir": "models/tseltal"
    }
}

def entrenar_y_guardar_modelos(idioma):
    config = IDIOMAS[idioma]
    print(f" Cargando dataset: {config['csv']}")
    df = pd.read_csv(config["csv"], encoding="utf-8")

    print(f" Entrenando modelo {idioma} → español...")
    vectorizer_idioma = TfidfVectorizer(analyzer='char', ngram_range=(1, 3))
    X_idioma = vectorizer_idioma.fit_transform(df[config["col_idioma"]])
    knn_idioma = NearestNeighbors(n_neighbors=1, metric="cosine")
    knn_idioma.fit(X_idioma)

    print(f" Entrenando modelo español → {idioma}...")
    vectorizer_esp = TfidfVectorizer(analyzer='char', ngram_range=(1, 3))
    X_esp = vectorizer_esp.fit_transform(df[config["col_espanol"]])
    knn_esp = NearestNeighbors(n_neighbors=1, metric="cosine")
    knn_esp.fit(X_esp)

    os.makedirs(config["modelo_dir"], exist_ok=True)

    print(f" Guardando modelos en carpeta '{config['modelo_dir']}'...")
    joblib.dump(df, os.path.join(config["modelo_dir"], "dataset.pkl"))
    joblib.dump(vectorizer_idioma, os.path.join(config["modelo_dir"], "vectorizer_zap.pkl"))
    joblib.dump(knn_idioma, os.path.join(config["modelo_dir"], "knn_zap.pkl"))
    joblib.dump(vectorizer_esp, os.path.join(config["modelo_dir"], "vectorizer_esp.pkl"))
    joblib.dump(knn_esp, os.path.join(config["modelo_dir"], "knn_esp.pkl"))

    print(f" Modelos de {idioma} entrenados y guardados correctamente.")

if __name__ == "__main__":
    print("INTERFAZ DE ENTRENAMIENTO DE MODELOS")
    print("1. Entrenar modelos zapoteco")
    print("2. Entrenar modelos tzeltal")
    print("3. Entrenar ambos idiomas")
    opcion = input("Selecciona una opción (1/2/3): ").strip()

    if opcion == "1":
        entrenar_y_guardar_modelos("zapoteco")
    elif opcion == "2":
        entrenar_y_guardar_modelos("tzeltal")
    elif opcion == "3":
        entrenar_y_guardar_modelos("zapoteco")
        entrenar_y_guardar_modelos("tzeltal")
    else:
        print("Opción no válida.")
