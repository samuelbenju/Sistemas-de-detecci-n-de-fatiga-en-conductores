# Sistema de detección de fatiga en conductores

## Tipo de aprendizaje utilizado

El proyecto emplea aprendizaje supervisado, ya que el modelo de inteligencia artificial fue entrenado con imágenes previamente etiquetadas en dos clases: `fatigado` y `despierto`.

Para la detección en tiempo real se utiliza un enfoque híbrido y secuencial que combina:

- Haar Cascade, para localizar el rostro en la imagen.
- MediaPipe Face Mesh, para calcular el EAR (Eye Aspect Ratio) y medir el nivel de apertura de los ojos.
- Red Neuronal Convolucional (CNN), para clasificar el rostro como fatigado o despierto.

Por tanto, el sistema no utiliza el modelo en cascada como metodología de desarrollo, sino una arquitectura de detección compuesta por varias técnicas complementarias.

Aplicación web desarrollada con **Flask**, **OpenCV**, **MediaPipe**, **TensorFlow/Keras**, **SQLite** y **Pygame** para la detección de fatiga en conductores mediante el análisis de la cámara web en tiempo real.

El sistema combina dos enfoques de detección:

1. **Modelo CNN** para clasificar el rostro como `fatigado` o `despierto`.
2. **Cálculo del EAR (Eye Aspect Ratio)** para detectar cierre prolongado de los ojos.

Además, incluye una interfaz web con autenticación por roles, gestión de imágenes para el dataset, reportes y almacenamiento de eventos en base de datos.

---

## Funcionalidades principales

- Inicio de sesión con roles `admin` y `cliente`.
- Panel de administración para:
  - Subir imágenes al dataset.
  - Eliminar imágenes del dataset.
  - Visualizar imágenes por categoría.
  - Generar reporte gráfico del dataset.
  - Descargar reporte en PDF.
- Panel de cliente.
- Módulo informativo.
- Módulo de comentarios.
- Módulo de contacto.
- Detección de fatiga en tiempo real con webcam.
- Activación de alarma sonora cuando se detecta fatiga.
- Registro de eventos de fatiga en base de datos SQLite.

---

## Tecnologías usadas

- **Python 3**
- **Flask**
- **SQLite3**
- **OpenCV**
- **MediaPipe**
- **TensorFlow / Keras**
- **NumPy**
- **SciPy**
- **Matplotlib**
- **ReportLab**
- **Pygame**

---

## Estructura sugerida del proyecto

```bash
proyecto/
│
├── inicio.py
├── deteccion_webcam_ear.py
├── proyecto.py
├── modificacion.py
├── kaggle.json
│
├── models/
│   └── mejor_modelo_fatiga.h5
│
├── static/
│   ├── alarma.mp3
│   └── imagenes/
│       ├── despierto/
│       └── fatigado/
│
├── templates/
│   ├── inicio.html
│   ├── login.html
│   ├── index.html
│   ├── cliente.html
│   ├── informacion.html
│   ├── comentarios.html
│   ├── contacto.html
│   └── reporte.html
│
└── usuarios.db
```

---

## Archivos del proyecto

### `inicio.py`
Archivo principal de la aplicación web.

Contiene:
- Configuración de Flask.
- Inicialización de la base de datos SQLite.
- Autenticación de usuarios.
- Panel de administrador y cliente.
- Registro de comentarios y contactos.
- Generación de reportes.
- Streaming de video desde la webcam.
- Detección de fatiga usando CNN + EAR.
- Activación y desactivación de la alarma.

### `deteccion_webcam_ear.py`
Script independiente para probar la detección de fatiga directamente desde la webcam, sin necesidad de ejecutar la aplicación web.

### `proyecto.py`
Script de entrenamiento del modelo de detección de fatiga.

Funciones principales:
- Configurar autenticación con Kaggle.
- Descargar dataset `Driver Drowsiness Dataset (DDD)`.
- Construir dataset balanceado.
- Integrar imágenes personalizadas.
- Entrenar el modelo CNN.
- Guardar el mejor modelo en `mejor_modelo_fatiga.h5`.

### `modificacion.py`
Script auxiliar para descargar y extraer un dataset adicional de rostros diversos desde Kaggle.

### `kaggle.json`
Archivo de credenciales de la API de Kaggle.

> **Importante:** este archivo no debe subirse a un repositorio público por motivos de seguridad.

---

## Requisitos previos

Antes de ejecutar el proyecto, asegúrate de tener instalado:

- Python 3.10 o superior
- pip
- Cámara web funcional
- Dependencias del proyecto

---

## Instalación

### 1. Clonar o descargar el proyecto

```bash
git clone <url-del-repositorio>
cd proyecto
```

### 2. Crear entorno virtual

```bash
python -m venv venv
```

### 3. Activar entorno virtual

**Windows:**

```bash
venv\Scripts\activate
```

**Linux / macOS:**

```bash
source venv/bin/activate
```

### 4. Instalar dependencias

```bash
pip install flask opencv-python mediapipe numpy scipy tensorflow matplotlib reportlab pygame kaggle
```

---

## Ejecución

### Ejecutar aplicación web

```bash
python inicio.py
```

La aplicación se ejecutará por defecto en:

```bash
http://127.0.0.1:5000/
```

### Ejecutar prueba local de detección por webcam

```bash
python deteccion_webcam_ear.py
```

### Entrenar el modelo

```bash
python proyecto.py
```

---

## Usuarios por defecto

El sistema crea automáticamente los siguientes usuarios:

| Usuario | Contraseña | Rol |
|---------|------------|-----|
| cristian.dominguez | 1014736792 | admin |
| erick.trujillo | 123 | admin |
| cliente01 | 456 | cliente |

> Se recomienda reemplazar estas credenciales por contraseñas seguras y almacenar hashes en vez de texto plano.

---

## Base de datos

La aplicación utiliza `SQLite` y crea automáticamente el archivo `usuarios.db` con las tablas:

- `usuarios`
- `log_fatiga`
- `comentarios`
- `contactos`

---

## Flujo general del sistema

1. El usuario inicia sesión.
2. Si es administrador, accede al panel de gestión del dataset y reportes.
3. Si es cliente, accede a su panel de usuario.
4. El sistema activa la cámara y procesa cada frame.
5. Se detecta fatiga mediante:
   - Predicción del modelo CNN.
   - Cálculo del EAR.
6. Si se detecta fatiga:
   - Se cambia el estado a `fatigado`.
   - Se activa una alarma sonora.
   - Se registra el evento en la base de datos.
7. Si el conductor vuelve a estar despierto:
   - Se desactiva la alarma.
   - Se registra el cambio de estado.

---

## Limitaciones actuales

- La contraseña de los usuarios se almacena en texto plano.
- La cámara se inicializa de manera global, lo que puede generar conflictos si hay varios accesos o si la cámara no está disponible.
- El archivo `kaggle.json` está expuesto.
- El sistema depende de que existan las carpetas, plantillas HTML, modelo `.h5` y audio `.mp3`.
- No se controla adecuadamente el cierre o liberación de la cámara al terminar.
- La función de logging intenta usar `session` dentro del generador de frames, lo cual puede generar problemas de contexto en algunos entornos Flask.

---

## Mejoras recomendadas

- Implementar hash de contraseñas con `werkzeug.security`.
- Mover `kaggle.json` fuera del repositorio.
- Crear un archivo `requirements.txt`.
- Separar la lógica de detección, base de datos y rutas en módulos distintos.
- Validar tamaño y extensión de imágenes subidas.
- Manejar correctamente la liberación de la cámara.
- Agregar manejo de errores cuando falte el modelo, el audio o la webcam.
- Proteger mejor las rutas administrativas.

---

## Estado del proyecto

El proyecto presenta una **base funcional a nivel de código** y una arquitectura coherente para una prueba académica o prototipo. Sin embargo, para considerarse completamente funcional en un entorno real, requiere:

- Plantillas HTML completas.
- Archivo del modelo entrenado.
- Archivo de audio de alarma.
- Validación de dependencias y entorno.
- Mejoras de seguridad y robustez.

En su estado actual, puede considerarse un **prototipo funcional condicionado**: la lógica está implementada, pero su ejecución completa depende de archivos y recursos adicionales no incluidos en este paquete.

---

## Licencia

Uso académico / prototipo.

