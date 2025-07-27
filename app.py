from flask import Flask, render_template, send_from_directory, request, redirect, url_for, flash, jsonify
from datetime import datetime
import os, csv, locale, shutil, secrets 

locale.setlocale(locale.LC_TIME, 'fr_FR.UTF-8')

app = Flask(__name__)
app.secret_key = secrets.token_hex(16)

VIDEO_DIR = "videos"
LOG_FILE = "logs/detections.csv"
LABEL_TRANSLATIONS = {
    "person": "Personne",
    "bird": "Oiseau"
}

def get_free_space_mb(path="/"):
    total, used, free = shutil.disk_usage(path)
    return free // (1024 * 1024)  # en Mo

def is_recording():
    return os.path.exists("/home/slaur/Documents/Birdwatcher/recording.flag")

@app.route('/')
def index():
    dates = sorted(os.listdir(VIDEO_DIR), reverse=True)
    videos_by_date = {}
    video_logs = {}
    
    free_space_mb = get_free_space_mb()
    low_disk_space = free_space_mb < 100

    log_entries = []
    if os.path.exists(LOG_FILE):
        with open(LOG_FILE, newline='') as f:
            reader = csv.reader(f)
            all_rows = list(reader)
            headers = all_rows[0]
            log_entries = all_rows[1:]

    for date in dates:
        path = os.path.join(VIDEO_DIR, date)
        if os.path.isdir(path):
            files = sorted(os.listdir(path), reverse=True)
            videos_by_date[date] = files
            for file in files:
                if file.endswith(".mp4"):
                    full_path = f"videos/{date}/{file}"
                    logs = [entry for entry in log_entries if entry[3] == full_path]
                    if logs:
                        raw = logs[0]
                        # Reformater les champs
                        timestamp = datetime.strptime(raw[0], "%Y-%m-%d_%H-%M-%S")
                        formatted_date = timestamp.strftime("%d %B %Y à %Hh%M")
                        label = LABEL_TRANSLATIONS.get(raw[1], raw[1].capitalize())
                        confidence = f"{float(raw[2]) * 100:.1f} %"
                        video_logs[file] = {
                            "date": formatted_date,
                            "label": label,
                            "confidence": confidence
                        }
                        
    return render_template("index.html", 
                           videos_by_date=videos_by_date, 
                           video_logs=video_logs, 
                           low_disk_space=low_disk_space, 
                           free_space_mb=free_space_mb, 
                           is_recording=is_recording()
                        )

@app.route('/videos/<date>/<filename>')
def serve_video(date, filename):
    return send_from_directory(os.path.join(VIDEO_DIR, date), filename)

@app.route("/delete", methods=["POST"])
def delete_video():
    video_rel_path = request.form.get("video_path")
    video_abs_path = os.path.join(VIDEO_DIR, video_rel_path)

    try:
        if os.path.exists(video_abs_path):
            os.remove(video_abs_path)
            flash(f"Vidéo supprimée : {video_rel_path}", "success")
        else:
            flash("Fichier introuvable.", "warning")
    except Exception as e:
        flash(f"Erreur lors de la suppression : {e}", "danger")

    return redirect(url_for("index"))

@app.route('/status')
def status():
    is_recording_status = is_recording()
    return jsonify({"is_recording": is_recording_status})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
