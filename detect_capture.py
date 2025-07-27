import RPi.GPIO as GPIO
import time, cv2, csv, os, subprocess
from picamera2 import Picamera2
from datetime import datetime
from picamera2.encoders import H264Encoder
from ultralytics import YOLO
import numpy as np

# === Configuration ===
PIR_GPIO = 14
IR_CUT_GPIO = 18
VIDEO_DIR = "videos"
MAX_DURATION = 45  # Durée maximale de chaque enregistrement
NO_PRESENCE_TIMEOUT = 6  # Durée sans détection avant arrêt
TARGET_CLASSES = ["person", "bird"]  # Classes à surveiller
STATUS_RECORDING_FILE = "/home/slaur/Documents/Birdwatcher/recording.flag"

# === Variables globales ===
last_check = 0
check_interval = 10  # secondes
is_night_mode = False

# === Création des répertoires nécessaires ===
os.makedirs(VIDEO_DIR, exist_ok=True)
today = datetime.now().strftime("%Y-%m-%d")
daily_video_dir = os.path.join(VIDEO_DIR, today)
os.makedirs(daily_video_dir, exist_ok=True)

os.makedirs("logs", exist_ok=True)

# === Initialisation des broches GPIO ===
GPIO.setmode(GPIO.BCM)
GPIO.setup(PIR_GPIO, GPIO.IN, pull_up_down=GPIO.PUD_DOWN)
GPIO.setup(IR_CUT_GPIO, GPIO.OUT)

# === Initialisation de la caméra et du système d'encodage ===
picam2 = Picamera2()
picam2.configure(picam2.create_video_configuration(main={"size": (1280, 720), "format": "RGB888"}))
encoder = H264Encoder()
picam2.start()

# === Chargement du modèle IA YOLOv8 léger ===
model = YOLO("yolov8n.pt")

# === Fonction pour démarrer l'enregistrement ===
def start_recording(target_detect, timestamp):
    set_recording_status(True)
    filename = f"{timestamp}_{target_detect}.h264"
    path = os.path.join(daily_video_dir, filename)
    print(f"Démarrage de l'enregistrement : {path}")
    picam2.start_recording(encoder, path)
    return path

# === Fonction pour arrêter l'enregistrement ===
def stop_recording(path, label=None, confidence=None, timestamp=None):
    path = path.replace(".h264", ".mp4")
    picam2.stop_recording()
    log_detection(label=label, confidence=confidence, video_path=path, timestamp=timestamp)
    set_recording_status(False)

# === Fonction de détection IA ===
def recognize_targets(frame):
    results = model(frame, verbose=False)
    boxes = results[0].boxes
    names = results[0].names

    for i in range(len(boxes.cls)):
        label = names[int(boxes.cls[i])]
        confidence = float(boxes.conf[i])
        if label in TARGET_CLASSES:
            print(f"Présence détectée : {label} ({confidence:.2f})")
            return label, confidence
    return None, None
    
# === Fonction pour enregistrer les détections dans un fichier CSV ===
def log_detection(timestamp, label, confidence, video_path=None):
    file_exists = os.path.isfile("logs/detections.csv")
    with open("logs/detections.csv", mode="a", newline="") as file:
        writer = csv.writer(file)
        if not file_exists or os.stat("logs/detections.csv").st_size == 0:
            writer.writerow(["timestamp", "label", "confidence", "video_path"])
        writer.writerow([timestamp, label, confidence, video_path])
        
# === Fonction pour convertir le fichier H264 en MP4 ===
def wrap_h264_to_mp4(h264_path):
    mp4_path = h264_path.replace(".h264", ".mp4")
    subprocess.run([
        "ffmpeg", "-y", "-framerate", "25",
        "-i", h264_path,
        "-c", "copy", mp4_path
    ])
    return mp4_path

# === Gestion du statut d'enregistrement ===
def set_recording_status(active: bool):
    if active:
        open(STATUS_RECORDING_FILE, "w").close()
    elif os.path.exists(STATUS_RECORDING_FILE):
        os.remove(STATUS_RECORDING_FILE)
        
# === Fonction pour vérifier si c'est un cadre de nuit ===        
def is_night_frame(frame, threshold=40):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    brightness = np.mean(gray)
    return brightness < threshold

# === Prétraitement des images pour améliorer la détection de nuit ===
def preprocess_frame(frame):
    if is_night_frame(frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        enhanced = clahe.apply(gray)
        return cv2.cvtColor(enhanced, cv2.COLOR_GRAY2BGR)
    else:
        return frame  # image de jour → pas de traitement

try:
    print("Système actif et prêt")
    while True:
        now = time.time()
        if now - last_check > check_interval:
            frame_check = picam2.capture_array()
            frame_check = cv2.cvtColor(frame_check, cv2.COLOR_BGRA2BGR)
            is_night_mode = is_night_frame(frame_check)
            last_check = now
            if is_night_mode:
                GPIO.output(IR_CUT_GPIO, GPIO.HIGH)
            else:
                GPIO.output(IR_CUT_GPIO, GPIO.LOW)

        if GPIO.input(PIR_GPIO) == GPIO.HIGH:
            print("Détection thermique : analyse IA en cours")
            frame = picam2.capture_array()
            frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)
            if is_night_mode:
                frame = preprocess_frame(frame)

            label, confidence = recognize_targets(frame)
            if label:
                timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
                video_path = start_recording(label, timestamp)
                start_time = time.time()
                last_presence_time = time.time()

                while time.time() - start_time < MAX_DURATION:
                    frame = picam2.capture_array()
                    frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)
                    if is_night_mode:
                        frame = preprocess_frame(frame)

                    label, conf = recognize_targets(frame)
                    if label:
                        last_presence_time = time.time()
                    elif time.time() - last_presence_time > NO_PRESENCE_TIMEOUT:
                        print("Aucune présence prolongée, arrêt anticipé")
                        break

                    time.sleep(0.05)

                stop_recording(video_path, label, confidence, timestamp)
                mp4_path = wrap_h264_to_mp4(video_path)
                os.remove(video_path)
                print(f"Vidéo convertie et sauvegardée : {mp4_path}")
                time.sleep(2)

        time.sleep(0.1)
    
except KeyboardInterrupt:
    print("Arrêt manuel demandé")

finally:
    picam2.stop()
    GPIO.cleanup()
