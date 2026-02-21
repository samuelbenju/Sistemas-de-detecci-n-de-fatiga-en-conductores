from kaggle.api.kaggle_api_extended import KaggleApi
import zipfile
import os

def descargar_faces_diversos():
    print("🔐 Autenticando con Kaggle API...")
    api = KaggleApi()
    api.authenticate()

    print("🔽 Descargando dataset...")
    api.dataset_download_files('sbaghbidi/human-faces-object-detection', path='.', unzip=False)

    print("📦 Descomprimiendo...")
    if os.path.exists("human-faces-object-detection.zip"):
        with zipfile.ZipFile("human-faces-object-detection.zip", 'r') as zip_ref:
            zip_ref.extractall("human_faces")
        print("✅ Listo: dataset extraído en carpeta 'human_faces'")
    else:
        print("❌ No se encontró el archivo ZIP descargado.")

descargar_faces_diversos()


