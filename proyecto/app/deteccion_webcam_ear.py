import cv2
import mediapipe as mp
import numpy as np
from scipy.spatial import distance
from tensorflow.keras.models import load_model
import pygame
import time

# --- Inicializar pygame (alarma) ---
pygame.mixer.init()
pygame.mixer.music.load("static/alarma.mp3")
alarma_sonando = False

# --- Cargar modelo CNN ---
modelo = load_model("models/mejor_modelo_fatiga.h5")

# --- MediaPipe ---
mp_face_detection = mp.solutions.face_detection
mp_drawing = mp.solutions.drawing_utils
mp_face_mesh = mp.solutions.face_mesh

face_detection = mp_face_detection.FaceDetection(min_detection_confidence=0.5)
face_mesh = mp_face_mesh.FaceMesh(refine_landmarks=True)

# --- Haarcascade para recorte de cara ---
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# --- Índices de ojos ---
LEFT_EYE = [33, 160, 158, 133, 153, 144]
RIGHT_EYE = [362, 385, 387, 263, 373, 380]

def eye_aspect_ratio(eye_landmarks, landmarks, w, h):
    points = [(int(landmarks[idx].x * w), int(landmarks[idx].y * h)) for idx in eye_landmarks]
    A = distance.euclidean(points[1], points[5])
    B = distance.euclidean(points[2], points[4])
    C = distance.euclidean(points[0], points[3])
    ear = (A + B) / (2.0 * C)
    return ear

# --- Parámetros ---
EAR_THRESHOLD = 0.25           # Ojos cerrados
CLOSED_FRAMES = 15             # Frames antes de contar fatiga por ojos
counter_ear = 0

UMBRAL_FATIGA = 0.75           # Umbral modelo CNN
DURACION_MINIMA_FATIGA = 2     # segundos
inicio_fatiga = None

cap = cv2.VideoCapture(0)
print("🎥 Cámara activa. Presiona 'q' para salir.")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    h, w, _ = frame.shape
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # --- MediaPipe detección y landmarks ---
    detections = face_detection.process(rgb_frame)
    results_mesh = face_mesh.process(rgb_frame)

    # --- CNN: detectar cara con Haarcascade ---
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=7, minSize=(60, 60))

    pred_fatiga = 0.0
    if len(faces) > 0:
        for (x, y, w_face, h_face) in faces:
            cara = frame[y:y+h_face, x:x+w_face]
            cara_resized = cv2.resize(cara, (150, 150))
            cara_normalizada = cara_resized / 255.0
            cara_reshaped = np.reshape(cara_normalizada, (1, 150, 150, 3))
            pred_fatiga = modelo.predict(cara_reshaped, verbose=0)[0][0]

            cv2.rectangle(frame, (x, y), (x + w_face, y + h_face), (255, 255, 255), 1)

    # --- EAR: apertura de ojos ---
    ear = 1.0
    if results_mesh.multi_face_landmarks:
        for face_landmarks in results_mesh.multi_face_landmarks:
            left_ear = eye_aspect_ratio(LEFT_EYE, face_landmarks.landmark, w, h)
            right_ear = eye_aspect_ratio(RIGHT_EYE, face_landmarks.landmark, w, h)
            ear = (left_ear + right_ear) / 2.0
            cv2.putText(frame, f"EAR: {ear:.2f}", (30, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    # --- Lógica de fatiga ---
    fatiga_por_ojos = False
    if ear < EAR_THRESHOLD:
        counter_ear += 1
        if counter_ear >= CLOSED_FRAMES:
            fatiga_por_ojos = True
    else:
        counter_ear = 0

    fatiga_por_modelo = pred_fatiga > UMBRAL_FATIGA

    # --- Determinar estado global ---
    if fatiga_por_ojos or fatiga_por_modelo:
        if inicio_fatiga is None:
            inicio_fatiga = time.time()
        elif time.time() - inicio_fatiga >= DURACION_MINIMA_FATIGA and not alarma_sonando:
            pygame.mixer.music.play(-1)
            alarma_sonando = True
        estado = " FATIGA DETECTADA"
        color_estado = (0, 0, 255)
    else:
        inicio_fatiga = None
        if alarma_sonando:
            pygame.mixer.music.stop()
            alarma_sonando = False
        estado = " Despierto"
        color_estado = (0, 255, 0)

    cv2.putText(frame, estado, (50, h - 30), cv2.FONT_HERSHEY_SIMPLEX, 1.2, color_estado, 3)

    cv2.imshow("Detección Fatiga Ojos + Cabeza", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
