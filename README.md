# Sistema de detección de fatiga en conductores

Aplicación web desarrollada con **Flask**, **OpenCV**, **MediaPipe**, **TensorFlow/Keras**, **SQLite** y **Pygame** para detectar fatiga en conductores mediante el análisis de la cámara web en tiempo real.

El sistema integra visión por computador y un modelo de inteligencia artificial para identificar señales de somnolencia o fatiga, generando alertas sonoras y registrando eventos en una base de datos local. Además, incorpora una interfaz web con autenticación por roles, administración del dataset, comentarios, contacto y generación de reportes.

---

## Descripción general

El proyecto fue diseñado como un prototipo funcional orientado a la prevención de accidentes por fatiga al conducir. Su funcionamiento principal consiste en capturar video desde la webcam, detectar el rostro del usuario, analizar el nivel de apertura ocular y clasificar el estado del conductor como **despierto** o **fatigado**.

La detección se realiza combinando tres técnicas:

1. **Haar Cascade**, para localizar el rostro dentro del frame.
2. **MediaPipe Face Mesh**, para obtener puntos faciales y calcular el **EAR (Eye Aspect Ratio)**.
3. **CNN (Red Neuronal Convolucional)**, para clasificar la imagen facial en las clases `fatigado` y `despierto`.

---

## Tipo de aprendizaje utilizado

El proyecto emplea **aprendizaje supervisado**, ya que el modelo de inteligencia artificial fue entrenado con imágenes previamente etiquetadas en dos clases:

- `fatigado`
- `despierto`

Durante el entrenamiento, el modelo aprende patrones visuales a partir de ejemplos clasificados para luego predecir el estado del conductor en tiempo real.

### ¿Usa modelo en cascada?

No utiliza el **modelo en cascada** como metodología de desarrollo de software.

Sin embargo, desde el punto de vista técnico, sí implementa una **detección híbrida y secuencial**, porque combina varias etapas de análisis:

- primero detecta el rostro,
- luego calcula el estado de los ojos,
- y finalmente usa una CNN para reforzar la clasificación.

Por eso, la forma correcta de describirlo es:

> **Sistema con aprendizaje supervisado y detección híbrida basada en visión por computador y red neuronal convolucional.**

---

## Funcionalidades principales

- Inicio de sesión con roles `admin` y `cliente`.
- Panel de administración para:
  - subir imágenes al dataset,
  - eliminar imágenes,
  - visualizar imágenes por categoría,
  - generar reportes gráficos,
  - descargar reportes en PDF.
- Panel de cliente.
- Módulo informativo.
- Módulo de comentarios.
- Módulo de contacto.
- Detección de fatiga en tiempo real mediante webcam.
- Alarma sonora cuando se detecta fatiga.
- Registro de eventos de fatiga en base de datos SQLite.
- Consulta del estado actual de fatiga desde una ruta web.

---

## Tecnologías utilizadas

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
- **Kaggle API**

---

## Estructura final del proyecto

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
- configuración de Flask,
- inicialización de la base de datos SQLite,
- autenticación de usuarios,
- panel de administrador y cliente,
- registro de comentarios y contactos,
- generación de reportes,
- streaming de video desde la webcam,
- detección de fatiga usando CNN + EAR,
- activación y desactivación de la alarma.

### `deteccion_webcam_ear.py`
Script independiente para probar la detección de fatiga directamente desde la webcam, sin necesidad de ejecutar la aplicación web completa.

### `proyecto.py`
Script encargado del entrenamiento del modelo de detección de fatiga.

Funciones principales:
- configurar autenticación con Kaggle,
- descargar el dataset `Driver Drowsiness Dataset (DDD)`,
- construir un dataset balanceado,
- integrar imágenes personalizadas,
- entrenar el modelo CNN,
- guardar el mejor modelo entrenado.

### `modificacion.py`
Script auxiliar para descargar y extraer un dataset adicional de rostros diversos desde Kaggle, con el fin de ampliar la variedad del conjunto de imágenes.

### `kaggle.json`
Archivo de credenciales de la API de Kaggle.

> **Importante:** este archivo no debe subirse a un repositorio público por razones de seguridad.

---

## Requisitos previos

Antes de ejecutar el proyecto, es necesario contar con:

- Python 3.10 o superior
- pip
- webcam funcional
- conexión a internet, si se van a descargar datasets desde Kaggle
- modelo entrenado `mejor_modelo_fatiga.h5`
- archivo de audio `alarma.mp3`
- plantillas HTML necesarias dentro de la carpeta `templates`

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

**Windows**

```bash
venv\Scripts\activate
```

**Linux / macOS**

```bash
source venv/bin/activate
```

### 4. Instalar dependencias

```bash
pip install flask opencv-python mediapipe numpy scipy tensorflow matplotlib reportlab pygame kaggle
```

---

## Ejecución del proyecto

### Ejecutar la aplicación web

```bash
python inicio.py
```

La aplicación se ejecuta, por defecto, en:

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

### Descargar dataset auxiliar de rostros

```bash
python modificacion.py
```

---

## Usuarios por defecto

El sistema crea automáticamente los siguientes usuarios:

| Usuario | Contraseña | Rol |
|---------|------------|-----|
| cristian.dominguez | 1014736792 | admin |
| erick.trujillo | 123 | admin |
| cliente01 | 456 | cliente |

> Se recomienda reemplazar estas credenciales por contraseñas seguras y almacenarlas con hash en lugar de texto plano.

---

## Base de datos

La aplicación utiliza `SQLite` y crea automáticamente el archivo `usuarios.db` con las siguientes tablas:

- `usuarios`
- `log_fatiga`
- `comentarios`
- `contactos`

Estas tablas permiten gestionar autenticación, almacenar eventos de fatiga y registrar información enviada por los usuarios desde la web.

---

## Flujo general del sistema

1. El usuario ingresa al sistema desde la interfaz web.
2. Inicia sesión con un rol de administrador o cliente.
3. Si es administrador, accede al panel de gestión del dataset y reportes.
4. Si es cliente, accede a su panel básico.
5. El sistema captura video desde la webcam.
6. En cada frame:
   - detecta el rostro,
   - calcula el EAR,
   - ejecuta la predicción del modelo CNN.
7. Si se detecta fatiga:
   - se cambia el estado a `fatigado`,
   - se activa una alarma sonora,
   - se registra el evento en la base de datos.
8. Si la persona vuelve al estado normal:
   - se desactiva la alarma,
   - se registra el cambio de estado.

---

## Estado actual del proyecto

El proyecto tiene estructura de **MVP funcional** y cuenta con componentes principales implementados. No obstante, para ejecutarlo correctamente es necesario disponer de todos los archivos complementarios, especialmente:

- plantillas HTML,
- modelo entrenado `.h5`,
- archivo de audio,
- carpetas del dataset,
- entorno con dependencias instaladas.

Por tanto, puede considerarse funcional a nivel académico y de prototipo, pero aún requiere ajustes para ser una solución robusta de producción.

---

## Posibles mejoras

- cifrar o hashear contraseñas de usuarios,
- separar la lógica en módulos para mejorar mantenimiento,
- agregar validaciones más estrictas de formularios y archivos,
- manejar mejor errores de cámara o ausencia de rostro,
- guardar métricas de desempeño del modelo,
- incorporar historial visual de eventos,
- desplegar la aplicación en un servidor web,
- agregar pruebas unitarias y documentación técnica adicional.

---

## Consideraciones de seguridad

- No subir `kaggle.json` a repositorios públicos.
- No dejar contraseñas en texto plano.
- No exponer `secret_key` fija en producción.
- Validar extensiones y tamaños de archivos subidos por los administradores.

---

## Conclusión

Este proyecto propone una solución tecnológica orientada a la seguridad vial mediante el uso de inteligencia artificial y visión por computador. Su mayor valor está en la integración de una interfaz web, una base de datos local y un sistema de detección de fatiga en tiempo real, lo cual lo convierte en un desarrollo adecuado para fines académicos, demostrativos y de investigación aplicada.
