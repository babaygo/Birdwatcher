import csv
import os
import subprocess
import time
from datetime import datetime

import cv2
import numpy as np
import RPi.GPIO as GPIO
from picamera2 import Picamera2
from picamera2.encoders import H264Encoder
from ultralytics import YOLO

# === Configuration ===
PIR_GPIO = 14
IR_CUT_GPIO = 18
VIDEO_DIR = "videos"
LOG_DIR = "logs"
MAX_DURATION = 45  # Durée max d'un enregistrement
NO_PRESENCE_TIMEOUT = 6  # Timeout avant arrêt si plus de détection
TARGET_CLASSES = {"person", "bird"}
STATUS_RECORDING_FILE = "/home/slaur/Documents/Birdwatcher/recording.flag"

CHECK_INTERVAL = 10  # secondes

# === Préparation des répertoires ===
os.makedirs(LOG_DIR, exist_ok=True)
today = datetime.now().strftime("%Y-%m-%d")
daily_video_dir = os.path.join(VIDEO_DIR, today)
os.makedirs(daily_video_dir, exist_ok=True)

# === GPIO ===
GPIO.setmode(GPIO.BCM)
GPIO.setup(PIR_GPIO, GPIO.IN, pull_up_down=GPIO.PUD_DOWN)
GPIO.setup(IR_CUT_GPIO, GPIO.OUT)

# === Caméra & encodeur ===
picam2 = Picamera2()
picam2.configure(
    picam2.create_video_configuration(main={"size": (1280, 720), "format": "RGB888"})
)
encoder = H264Encoder()
picam2.start()

# === IA YOLO ===
model = YOLO("yolov8n.pt")


# --- Fonctions utilitaires ---
def set_recording_status(active: bool):
    """Crée ou supprime le flag indiquant un enregistrement en cours."""
    if active:
        open(STATUS_RECORDING_FILE, "w").close()
    elif os.path.exists(STATUS_RECORDING_FILE):
        os.remove(STATUS_RECORDING_FILE)


def capture_frame():
    """Capture une image de la caméra au format BGR."""
    return cv2.cvtColor(picam2.capture_array(), cv2.COLOR_BGRA2BGR)


def is_night_frame(frame, threshold=40):
    """Retourne True si le frame est considéré comme de nuit (faible luminosité)."""
    return np.mean(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)) < threshold


def preprocess_frame(frame, night: bool):
    """Améliore les images nocturnes avec CLAHE."""
    if not night:
        return frame
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    return cv2.cvtColor(clahe.apply(gray), cv2.COLOR_GRAY2BGR)


def recognize_targets(frame):
    """Retourne le (label, confidence) s’il y a une détection d’intérêt."""
    results = model(frame, verbose=False)
    boxes, names = results[0].boxes, results[0].names
    for cls, conf in zip(boxes.cls, boxes.conf):
        label = names[int(cls)]
        if label in TARGET_CLASSES:
            print(f"Présence détectée : {label} ({conf:.2f})")
            return label, float(conf)
    return None, None


def log_detection(timestamp, label, confidence, video_path=None):
    """Ajoute une ligne dans le CSV des détections."""
    log_path = os.path.join(LOG_DIR, "detections.csv")
    file_exists = os.path.isfile(log_path)

    with open(log_path, "a", newline="") as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(["timestamp", "label", "confidence", "video_path"])
        writer.writerow([timestamp, label, confidence, video_path])


def start_recording(target_detect, timestamp):
    """Démarre l’enregistrement vidéo."""
    set_recording_status(True)
    filename = f"{timestamp}_{target_detect}.h264"
    path = os.path.join(daily_video_dir, filename)
    print(f"Enregistrement démarré : {path}")
    picam2.start_recording(encoder, path)
    return path


def stop_recording(h264_path, label=None, confidence=None, timestamp=None):
    """Arrête l’enregistrement, convertit en MP4, réinitialise la caméra et log la détection."""
    picam2.stop_recording()

    mp4_path = h264_path.replace(".h264", ".mp4")
    subprocess.run(
        ["ffmpeg", "-y", "-framerate", "25", "-i", h264_path, "-c", "copy", mp4_path],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    os.remove(h264_path)

    log_detection(timestamp, label, confidence, mp4_path)
    set_recording_status(False)
    print(f"Vidéo sauvegardée : {mp4_path}")

    picam2.stop()
    picam2.configure(
        picam2.create_video_configuration(
            main={"size": (1280, 720), "format": "RGB888"}
        )
    )
    picam2.start()


# --- Boucle principale ---
try:
    print("Système actif et prêt")
    last_check, night_mode = 0, False

    while True:
        now = time.time()

        # Mise à jour mode jour/nuit périodiquement
        if now - last_check > CHECK_INTERVAL:
            frame = capture_frame()
            night_mode = is_night_frame(frame)
            GPIO.output(IR_CUT_GPIO, GPIO.HIGH if night_mode else GPIO.LOW)
            last_check = now

        # Détection PIR
        if GPIO.input(PIR_GPIO) == GPIO.HIGH:
            print("Détection thermique : analyse IA en cours")
            frame = preprocess_frame(capture_frame(), night_mode)
            label, confidence = recognize_targets(frame)

            if label:  # Si cible détectée
                timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
                video_path = start_recording(label, timestamp)

                start_time = time.time()
                last_presence = start_time

                # Boucle d'enregistrement
                while time.time() - start_time < MAX_DURATION:
                    frame = preprocess_frame(capture_frame(), night_mode)
                    lbl, _ = recognize_targets(frame)

                    if lbl:
                        last_presence = time.time()
                    elif time.time() - last_presence > NO_PRESENCE_TIMEOUT:
                        print("Aucune présence prolongée, arrêt anticipé")
                        break

                    time.sleep(0.05)

                stop_recording(video_path, label, confidence, timestamp)
                time.sleep(2)  # petite pause avant reprise

        time.sleep(0.1)

except KeyboardInterrupt:
    print("Arrêt manuel demandé")

finally:
    picam2.stop()
    GPIO.cleanup()
