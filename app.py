import csv
import locale
import os
import secrets
import shutil
import zipfile
import io
from datetime import datetime

from flask import (
    Flask,
    flash,
    redirect,
    render_template,
    request,
    send_from_directory,
    url_for,
    send_file,
)

locale.setlocale(locale.LC_TIME, "fr_FR.UTF-8")

app = Flask(__name__)
app.secret_key = secrets.token_hex(16)

VIDEO_DIR = "videos"
LOG_FILE = "logs/detections.csv"
LABEL_TRANSLATIONS = {"person": "Personne", "bird": "Oiseau"}


def load_detections():
    detections = {}
    with open("logs/detections.csv", newline="") as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            video_path = row["video_path"]

            rel_path = os.path.relpath(video_path, start=VIDEO_DIR)

            label = row["label"]
            confidence = float(row["confidence"])
            timestamp_str = row.get("timestamp", "")

            try:
                dt = datetime.strptime(timestamp_str, "%Y-%m-%d_%H-%M-%S")
                formatted_date = dt.strftime("%d %B %Y à %Hh%M")
            except ValueError:
                formatted_date = "Date inconnue"

            translated_label = LABEL_TRANSLATIONS.get(label, label)

            detections[rel_path] = {
                "label": translated_label,
                "confidence": round(confidence * 100, 1),
                "date": formatted_date,
            }
    return detections


def get_stockage():
    total, used, free = shutil.disk_usage("/")
    return total // (2**20), used // (2**20), free // (2**20)


# Routes
@app.route("/")
def index():
    total_memoire, used_memoire, memoire = get_stockage()
    return render_template(
        "index.html",
        memoire=memoire,
        used_memoire=used_memoire,
        total_memoire=total_memoire,
    )


@app.route("/videos")
def videos():
    dates = sorted(os.listdir(VIDEO_DIR), reverse=True)
    detections = load_detections()
    video_map = {}

    for date in dates:
        path = os.path.join(VIDEO_DIR, date)
        if os.path.isdir(path):
            files = sorted(os.listdir(path), reverse=True)
            video_files = [f for f in files if f.lower().endswith((".mp4"))]
            if video_files:
                video_map[date] = video_files

    return render_template("videos.html", video_map=video_map, detections=detections)


@app.route("/videos/<date>/<filename>")
def serve_video(date, filename):
    return send_from_directory(os.path.join(VIDEO_DIR, date), filename)


@app.route("/delete", methods=["POST"])
def delete_video():
    video_paths = request.form.getlist("video_paths")

    if not video_paths:
        single_path = request.form.get("video_path")
        if single_path:
            video_paths = [single_path]

    if not video_paths:
        flash("Aucune vidéo sélectionnée.", "warning")
        return redirect(url_for("videos"))

    deleted_count = 0
    errors = []

    for video_rel_path in video_paths:
        video_abs_path = os.path.abspath(os.path.join(VIDEO_DIR, video_rel_path))

        if not video_abs_path.startswith(os.path.abspath(VIDEO_DIR)):
            errors.append(f"{video_rel_path}: Chemin non autorisé")
            continue

        try:
            if os.path.exists(video_abs_path):
                os.remove(video_abs_path)
                deleted_count += 1
            else:
                errors.append(f"{video_rel_path}: Fichier introuvable")
        except Exception as e:
            errors.append(f"{video_rel_path}: {str(e)}")

    # Messages de retour
    if deleted_count > 0:
        if deleted_count == 1:
            flash("1 vidéo supprimée avec succès.", "success")
        else:
            flash(f"{deleted_count} vidéos supprimées avec succès.", "success")

    if errors:
        for error in errors:
            flash(error, "danger")

    return redirect(url_for("videos"))


@app.route("/download-videos-zip", methods=["POST"])
def download_videos_zip():
    data = request.get_json()
    video_paths = data.get("video_paths", [])

    if not video_paths:
        return {"error": "Aucune vidéo sélectionnée"}, 400

    # Créer un ZIP en mémoire
    memory_file = io.BytesIO()

    with zipfile.ZipFile(memory_file, "w", zipfile.ZIP_DEFLATED) as zipf:
        for video_path in video_paths:
            date, filename = video_path.split("/")
            full_path = os.path.join("videos", date, filename)

            if os.path.exists(full_path):
                # Ajouter au ZIP avec un nom simplifié
                zipf.write(full_path, arcname=f"{date}_{filename}")

    memory_file.seek(0)

    return send_file(
        memory_file,
        mimetype="application/zip",
        as_attachment=True,
        download_name=f"birdwatcher_videos_{date}.zip",
    )


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
