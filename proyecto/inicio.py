from flask import Flask, render_template, request, redirect, url_for, session, send_file, flash, Response
from werkzeug.utils import secure_filename
import sqlite3
import os
import io
import base64
import matplotlib.pyplot as plt
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import letter
from datetime import datetime
import cv2
import numpy as np
import mediapipe as mp
from scipy.spatial import distance
from tensorflow.keras.models import load_model
import time
import pygame

pygame.mixer.init()
pygame.mixer.music.load("static/alarma.mp3")
alarma_sonando = False
estado_fatiga = {"estado": "despierto"}

# -------------------------
# CONFIGURACIÓN FLASK
# -------------------------
app = Flask(__name__)
app.secret_key = "clave_secreta"

UPLOAD_FOLDER = os.path.join("static", "imagenes")
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
for clase in ["despierto", "fatigado"]:
    os.makedirs(os.path.join(UPLOAD_FOLDER, clase), exist_ok=True)

# ------------------------- 
# BASE DE DATOS 
# ------------------------- 
def init_db():
    conn = sqlite3.connect("usuarios.db")
    cursor = conn.cursor()
    
    # Tabla de usuarios
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS usuarios (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT UNIQUE NOT NULL,
            password TEXT NOT NULL,
            rol TEXT NOT NULL
        )
    ''')
    
    # Tabla de logs de fatiga
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS log_fatiga (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT,
            evento TEXT NOT NULL,
            ear REAL,
            prob REAL,
            ts TEXT NOT NULL DEFAULT (datetime('now'))
        )
    """)
    
    # Tabla de comentarios
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS comentarios (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            nombre TEXT NOT NULL,
            email TEXT NOT NULL,
            comentario TEXT NOT NULL,
            fecha TEXT NOT NULL
        )
    ''')
    
    # Tabla de contactos
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS contactos (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            nombre TEXT NOT NULL,
            email TEXT NOT NULL,
            asunto TEXT NOT NULL,
            mensaje TEXT NOT NULL,
            fecha TEXT NOT NULL
        )
    ''')
    
    conn.commit()
    
    # Insertar usuarios por defecto
    try:
        cursor.execute("INSERT INTO usuarios (username, password, rol) VALUES (?, ?, ?)", 
                      ("cristian.dominguez", "1014736792", "admin"))
        cursor.execute("INSERT INTO usuarios (username, password, rol) VALUES (?, ?, ?)", 
                      ("erick.trujillo", "123", "admin"))
        cursor.execute("INSERT INTO usuarios (username, password, rol) VALUES (?, ?, ?)", 
                      ("cliente01", "456", "cliente"))
        conn.commit()
    except sqlite3.IntegrityError:
        pass  # Los usuarios ya existen
    
    conn.close()

init_db()

# --- NUEVO: helper de logging + estado previo ---
def _log_event(evento, ear=None, prob=None):
    """
    Inserta un evento en log_fatiga sin romper el streaming si algo falla.
    evento: 'fatigado' | 'despierto' | 'alarma_on' | 'alarma_off'
    """
    try:
        con = sqlite3.connect("usuarios.db")
        cur = con.cursor()
        username = None
        try:
            username = session.get("usuario")
        except Exception:
            pass
        cur.execute("""
            INSERT INTO log_fatiga (username, evento, ear, prob)
            VALUES (?, ?, ?, ?)
        """, (username, str(evento),
              float(ear) if ear is not None else None,
              float(prob) if prob is not None else None))
        con.commit()
    except Exception as e:
        print("WARN log_fatiga:", e)
    finally:
        try:
            con.close()
        except:
            pass

prev_estado = "despierto"
# --- /NUEVO ---

# -------------------------
# MODELO Y DETECTORES
# -------------------------
modelo = load_model("models/mejor_modelo_fatiga.h5")
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(refine_landmarks=True)

LEFT_EYE = [33, 160, 158, 133, 153, 144]
RIGHT_EYE = [362, 385, 387, 263, 373, 380]

def eye_aspect_ratio(eye_landmarks, landmarks, w, h):
    points = [(int(landmarks[idx].x * w), int(landmarks[idx].y * h)) for idx in eye_landmarks]
    A = distance.euclidean(points[1], points[5])
    B = distance.euclidean(points[2], points[4])
    C = distance.euclidean(points[0], points[3])
    ear = (A + B) / (2.0 * C)
    return ear

EAR_THRESHOLD = 0.25
CLOSED_FRAMES = 15
counter_ear = 0

# -------------------------
# RUTAS FLASK
# -------------------------
@app.route("/")
def inicio():
    return render_template("inicio.html")

@app.route("/login", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        username = request.form["username"]
        password = request.form["password"]

        conn = sqlite3.connect("usuarios.db")
        cursor = conn.cursor()
        cursor.execute("SELECT rol FROM usuarios WHERE username=? AND password=?", (username, password))
        user = cursor.fetchone()
        conn.close()

        if user:
            rol = user[0]
            session["usuario"] = username
            session["rol"] = rol
            if rol == "admin":
                return redirect(url_for("admin_index"))
            elif rol == "cliente":
                return redirect(url_for("cliente"))
        else:
            return render_template("login.html", error="Usuario o contraseña incorrectos")
    return render_template("login.html")

@app.route("/logout")
def logout():
    session.clear()
    return redirect(url_for("inicio"))

# -------------------------
# PANEL ADMIN
# -------------------------
@app.route("/admin")
def admin_index():
    if "rol" not in session or session["rol"] != "admin":
        return redirect(url_for("inicio"))

    despierto_path = os.path.join(app.static_folder, "imagenes/despierto")
    fatigado_path = os.path.join(app.static_folder, "imagenes/fatigado")

    imagenes_despierto = os.listdir(despierto_path)
    imagenes_fatigado = os.listdir(fatigado_path)

    return render_template("index.html",
                           usuario=session["usuario"],
                           imagenes_despierto=imagenes_despierto,
                           imagenes_fatigado=imagenes_fatigado)

@app.route("/upload", methods=["POST"])
def upload():
    if "rol" not in session or session["rol"] != "admin":
        return redirect(url_for("inicio"))

    imagen = request.files["imagen"]
    clase = request.form["categoria"]
    if imagen and clase:
        filename = secure_filename(imagen.filename)
        ruta = os.path.join(app.config["UPLOAD_FOLDER"], clase, filename)
        imagen.save(ruta)
        flash("✅ Imagen subida correctamente")
    return redirect(url_for("admin_index"))

@app.route("/eliminar/<clase>/<nombre>")
def eliminar_imagen(clase, nombre):
    if "rol" not in session or session["rol"] != "admin":
        return redirect(url_for("inicio"))
    ruta = os.path.join(app.config["UPLOAD_FOLDER"], clase, nombre)
    if os.path.exists(ruta):
        os.remove(ruta)
        flash("🗑️ Imagen eliminada")
    return redirect(url_for("admin_index"))

@app.route("/reporte")
def reporte():
    if "rol" not in session or session["rol"] != "admin":
        return redirect(url_for("inicio"))

    clases = ["despierto", "fatigado"]
    conteos = {}
    total = 0

    for clase in clases:
        ruta = os.path.join(UPLOAD_FOLDER, clase)
        num_imgs = len(os.listdir(ruta))
        conteos[clase] = num_imgs
        total += num_imgs

    porcentajes = {k: (v / total * 100) if total > 0 else 0 for k, v in conteos.items()}

    fig, ax = plt.subplots()
    ax.bar(conteos.keys(), conteos.values(), color=["green", "red"])
    ax.set_title("Cantidad de Imágenes por Clase")
    ax.set_ylabel("Número de imágenes")

    img = io.BytesIO()
    plt.savefig(img, format="png")
    img.seek(0)
    img_base64 = base64.b64encode(img.getvalue()).decode("utf8")
    plt.close()

    return render_template("reporte.html", conteos=conteos, porcentajes=porcentajes, grafico=img_base64)

@app.route("/reporte/pdf")
def reporte_pdf():
    if "rol" not in session or session["rol"] != "admin":
        return redirect(url_for("inicio"))

    despierto = len(os.listdir(os.path.join(UPLOAD_FOLDER, "despierto")))
    fatigado = len(os.listdir(os.path.join(UPLOAD_FOLDER, "fatigado")))
    total = despierto + fatigado

    buffer = io.BytesIO()
    c = canvas.Canvas(buffer, pagesize=letter)
    width, height = letter

    c.setFont("Helvetica-Bold", 18)
    c.drawCentredString(width / 2, height - 50, "📊 Reporte del Dataset de Fatiga")

    c.setFont("Helvetica", 12)
    y = height - 100
    c.drawString(100, y, f"Total de imágenes en el dataset: {total}")
    y -= 30
    c.drawString(100, y, f"- Imágenes etiquetadas como 'Despierto': {despierto}")
    y -= 30
    c.drawString(100, y, f"- Imágenes etiquetadas como 'Fatigado': {fatigado}")
    y -= 50
    c.drawString(100, y, "Fecha de generación del reporte: ")
    c.drawString(300, y, datetime.now().strftime("%Y-%m-%d %H:%M:%S"))

    c.showPage()
    c.save()

    buffer.seek(0)
    return send_file(buffer, as_attachment=True,
                     download_name="reporte_dataset.pdf",
                     mimetype="application/pdf")
    
    

# -------------------------
# PANEL CLIENTE
# -------------------------
@app.route("/cliente")
def cliente():
    if "rol" in session and session["rol"] == "cliente":
        return render_template("cliente.html", usuario=session["usuario"])
    return redirect(url_for("inicio"))


# ------------------------- 
# MÓDULO DE INFORMACIÓN
# ------------------------- 
@app.route("/informacion")
def informacion():
    return render_template("informacion.html")

# ------------------------- 
# MÓDULO DE COMENTARIOS
# ------------------------- 
@app.route("/comentarios", methods=["GET", "POST"])
def comentarios():
    if request.method == "POST":
        nombre = request.form.get("nombre")
        email = request.form.get("email")
        comentario = request.form.get("comentario")
        
        # Guardar comentario en la base de datos
        try:
            conn = sqlite3.connect("usuarios.db")
            cursor = conn.cursor()
            cursor.execute("""
                INSERT INTO comentarios (nombre, email, comentario, fecha)
                VALUES (?, ?, ?, ?)
            """, (nombre, email, comentario, datetime.now().strftime("%Y-%m-%d %H:%M:%S")))
            conn.commit()
            conn.close()
            flash("✅ Comentario enviado correctamente")
        except Exception as e:
            flash("❌ Error al enviar el comentario")
            print(f"Error: {e}")
        
        return redirect(url_for("comentarios"))
    
    # Mostrar todos los comentarios
    conn = sqlite3.connect("usuarios.db")
    cursor = conn.cursor()
    cursor.execute("SELECT nombre, comentario, fecha FROM comentarios ORDER BY fecha DESC")
    comentarios = cursor.fetchall()
    conn.close()
    
    return render_template("comentarios.html", comentarios=comentarios)

# ------------------------- 
# MÓDULO DE CONTACTO
# ------------------------- 
@app.route("/contacto", methods=["GET", "POST"])
def contacto():
    if request.method == "POST":
        nombre = request.form.get("nombre")
        email = request.form.get("email")
        asunto = request.form.get("asunto")
        mensaje = request.form.get("mensaje")
        
        # Guardar mensaje de contacto en la base de datos
        try:
            conn = sqlite3.connect("usuarios.db")
            cursor = conn.cursor()
            cursor.execute("""
                INSERT INTO contactos (nombre, email, asunto, mensaje, fecha)
                VALUES (?, ?, ?, ?, ?)
            """, (nombre, email, asunto, mensaje, datetime.now().strftime("%Y-%m-%d %H:%M:%S")))
            conn.commit()
            conn.close()
            flash("✅ Mensaje enviado correctamente. Te contactaremos pronto.")
        except Exception as e:
            flash("❌ Error al enviar el mensaje")
            print(f"Error: {e}")
        
        return redirect(url_for("contacto"))
    
    return render_template("contacto.html")

# -------------------------
# STREAMING DETECTOR
# -------------------------
camera = cv2.VideoCapture(0)

def generar_frames():
    global counter_ear, prev_estado, alarma_sonando
    while True:
        success, frame = camera.read()
        if not success:
            break

        h, w, _ = frame.shape
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # CNN con Haarcascade
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=6, minSize=(60, 60))
        pred = 0.0
        for (x, y, w_face, h_face) in faces:
            cara = frame[y:y+h_face, x:x+w_face]
            cara_resized = cv2.resize(cara, (150, 150))
            cara_normalizada = cara_resized / 255.0
            cara_reshaped = np.reshape(cara_normalizada, (1, 150, 150, 3))
            pred = modelo.predict(cara_reshaped, verbose=0)[0][0]
            cv2.rectangle(frame, (x, y), (x + w_face, y + h_face), (255, 255, 255), 1)

        # EAR con MediaPipe
        results_mesh = face_mesh.process(rgb_frame)
        ear = 1.0
        if results_mesh.multi_face_landmarks:
            for face_landmarks in results_mesh.multi_face_landmarks:
                left_ear = eye_aspect_ratio(LEFT_EYE, face_landmarks.landmark, w, h)
                right_ear = eye_aspect_ratio(RIGHT_EYE, face_landmarks.landmark, w, h)
                ear = (left_ear + right_ear) / 2.0
                cv2.putText(frame, f"EAR: {ear:.2f}", (30, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        # Estado global
        fatiga_por_ojos = False
        if ear < EAR_THRESHOLD:
            counter_ear += 1
            if counter_ear >= CLOSED_FRAMES:
                fatiga_por_ojos = True
        else:
            counter_ear = 0

        fatiga_por_modelo = pred > 0.75

        if fatiga_por_ojos or fatiga_por_modelo:
            estado = "FATIGADO"
            color = (0, 0, 255)
            estado_fatiga["estado"] = "fatigado"
        else:
            estado = "DESPIERTO"
            color = (0, 255, 0)
            estado_fatiga["estado"] = "despierto"

        # --- NUEVO: log de cambios de estado ---
        if estado_fatiga["estado"] != prev_estado:
            _log_event(estado_fatiga["estado"], ear=ear, prob=pred)
            prev_estado = estado_fatiga["estado"]

        # --- NUEVO: control/log de alarma ---
        if estado_fatiga["estado"] == "fatigado" and not alarma_sonando:
            try:
                pygame.mixer.music.play(-1)  # bucle
                alarma_sonando = True
                _log_event("alarma_on", ear=ear, prob=pred)
            except Exception as e:
                print("WARN alarma_on:", e)

        if estado_fatiga["estado"] == "despierto" and alarma_sonando:
            try:
                pygame.mixer.music.stop()
                alarma_sonando = False
                _log_event("alarma_off", ear=ear, prob=pred)
            except Exception as e:
                print("WARN alarma_off:", e)

        cv2.putText(frame, estado, (50, h - 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, color, 3)

        _, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/video_feed')
def video_feed():
    return Response(generar_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')
    
    
estado_fatiga = {"estado": "despierto"}

@app.route("/estado_fatiga")
def estado_fatiga_route():
    return estado_fatiga


# -------------------------
# MAIN
# -------------------------
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
