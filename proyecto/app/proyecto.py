import os
import zipfile
import shutil
import glob
from kaggle.api.kaggle_api_extended import KaggleApi
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.callbacks import ModelCheckpoint
import matplotlib.pyplot as plt

# ---------------- CONFIGURACIÓN INICIAL ----------------

KAGGLE_JSON_PATH = "kaggle.json"
DATASET_NAME = "ismailnasri20/driver-drowsiness-dataset-ddd"
DATASET_ZIP_NAME = "driver-drowsiness-dataset-ddd.zip"
EXTRACTION_PATH = "dataset"
FACES_DATASET_PATH = "human_faces/images"
BALANCED_DATASET_PATH = "dataset_balanceado"

# ---------------- AUTENTICACIÓN KAGGLE ----------------

def configurar_kaggle():
    kaggle_dir = os.path.expanduser("~/.kaggle")
    os.makedirs(kaggle_dir, exist_ok=True)
    shutil.copy(KAGGLE_JSON_PATH, os.path.join(kaggle_dir, "kaggle.json"))
    os.chmod(os.path.join(kaggle_dir, "kaggle.json"), 0o600)

# ---------------- DESCARGAR Y EXTRAER DATASET DDD ----------------

def descargar_y_extraer_dataset():
    api = KaggleApi()
    api.authenticate()
    api.dataset_download_files(dataset=DATASET_NAME, path=".", unzip=False)
    print("📦 Extrayendo dataset DDD...")
    with zipfile.ZipFile(DATASET_ZIP_NAME, 'r') as zip_ref:
        zip_ref.extractall(EXTRACTION_PATH)

# ---------------- CONSTRUIR DATASET BALANCEADO ----------------

def construir_dataset_balanceado():
    print("🧩 Combinando datasets para incluir diversidad...")

    origen_ddd = os.path.join(EXTRACTION_PATH, "Driver Drowsiness Dataset (DDD)")
    destino_base = BALANCED_DATASET_PATH

    clases_origen_destino = {
        "Drowsy": "fatigado",
        "Non Drowsy": "despierto"
    }

    for origen, destino in clases_origen_destino.items():
        origen_clase = os.path.join(origen_ddd, origen)
        destino_clase = os.path.join(destino_base, destino)
        os.makedirs(destino_clase, exist_ok=True)

        for img in glob.glob(os.path.join(origen_clase, "*.png")):
            shutil.copy(img, destino_clase)

    # Agregar rostros diversos como "despierto"
    destino_despierto = os.path.join(destino_base, 'despierto')
    faces = glob.glob(os.path.join(FACES_DATASET_PATH, "*.png"))[:100]

    for i, img_path in enumerate(faces):
        nombre = f"human_{i}.png"
        shutil.copy(img_path, os.path.join(destino_despierto, nombre))

    print("✅ Dataset balanceado creado en carpeta 'dataset_balanceado'")

# ---------------- INTEGRAR ROSTRO PERSONAL ----------------

def integrar_rostro_personal():
    print("➕ Integrando imágenes personalizadas desde 'rostro_dataset_v4'...")

    personal_dataset = "rostro_dataset_v4"
    for clase in ["despierto", "fatigado"]:
        origen_clase = os.path.join(personal_dataset, clase)
        destino_clase = os.path.join(BALANCED_DATASET_PATH, clase)

        os.makedirs(destino_clase, exist_ok=True)

        imagenes = glob.glob(os.path.join(origen_clase, "*.png"))
        for i, ruta in enumerate(imagenes):
            nombre_archivo = f"rostro_personal_{clase}_{i+1:03}.png"
            shutil.copy(ruta, os.path.join(destino_clase, nombre_archivo))

        print(f"✅ Copiadas {len(imagenes)} imágenes de '{clase}' al dataset balanceado.")

# ---------------- CARGAR Y PREPROCESAR DATOS ----------------

def cargar_datos():
    train_dir = BALANCED_DATASET_PATH
    img_size = (150, 150)
    batch_size = 32

    train_datagen = ImageDataGenerator(
        rescale=1./255,
        validation_split=0.2,
        rotation_range=20,
        zoom_range=0.2,
        horizontal_flip=True
    )

    train_generator = train_datagen.flow_from_directory(
        train_dir,
        target_size=img_size,
        batch_size=batch_size,
        class_mode='binary',
        subset='training'
    )

    val_generator = train_datagen.flow_from_directory(
        train_dir,
        target_size=img_size,
        batch_size=batch_size,
        class_mode='binary',
        subset='validation'
    )

    return train_generator, val_generator

# ---------------- DEFINIR Y ENTRENAR MODELO ----------------

def entrenar_modelo(train_gen, val_gen):
    model = Sequential([
        Conv2D(32, (3,3), activation='relu', input_shape=(150, 150, 3)),
        MaxPooling2D(2,2),
        Conv2D(64, (3,3), activation='relu'),
        MaxPooling2D(2,2),
        Conv2D(128, (3,3), activation='relu'),
        MaxPooling2D(2,2),
        Flatten(),
        Dense(512, activation='relu'),
        Dropout(0.5),
        Dense(1, activation='sigmoid')
    ])

    model.compile(
        loss='binary_crossentropy',
        optimizer='adam',
        metrics=['accuracy']
    )

    model.summary()

    checkpoint = ModelCheckpoint(
        "mejor_modelo_fatiga.h5",
        monitor='val_accuracy',
        save_best_only=True,
        mode='max',
        verbose=1
    )

    history = model.fit(
        train_gen,
        validation_data=val_gen,
        epochs=20,
        callbacks=[checkpoint]
    )

    plot_history(history)
    print("✅ Entrenamiento finalizado. El mejor modelo se guardó como 'mejor_modelo_fatiga.h5'")

# ---------------- VISUALIZAR MÉTRICAS ----------------

def plot_history(history):
    plt.plot(history.history['accuracy'], label='train_acc')
    plt.plot(history.history['val_accuracy'], label='val_acc')
    plt.title("Precisión del Modelo")
    plt.legend()
    plt.show()

    plt.plot(history.history['loss'], label='train_loss')
    plt.plot(history.history['val_loss'], label='val_loss')
    plt.title("Pérdida del Modelo")
    plt.legend()
    plt.show()

# ---------------- MAIN ----------------

if __name__ == "__main__":
    configurar_kaggle()
    descargar_y_extraer_dataset()
    construir_dataset_balanceado()
    integrar_rostro_personal()
    train_gen, val_gen = cargar_datos()
    entrenar_modelo(train_gen, val_gen)
