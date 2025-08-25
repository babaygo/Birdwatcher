import csv
import locale
import os
import secrets
from datetime import datetime

from flask import (
    Flask,
    flash,
    redirect,
    render_template,
    request,
    send_from_directory,
    url_for,
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
            video = row["video_path"]
            label = row["label"]
            confidence = float(row["confidence"])
            timestamp_str = row.get("timestamp", "")
            try:
                dt = datetime.strptime(timestamp_str, "%Y-%m-%d_%H-%M-%S")
                formatted_date = dt.strftime("%d %B %Y à %Hh%M")
            except ValueError:
                formatted_date = "Date inconnue"

            translated_label = LABEL_TRANSLATIONS.get(label, label)
            detections[video] = {
                "label": translated_label,
                "confidence": round(confidence * 100, 1),
                "date": formatted_date,
            }
    return detections


# Routes
@app.route("/")
def index():
    return render_template("index.html")


@app.route("/videos")
def videos():
    dates = sorted(os.listdir(VIDEO_DIR), reverse=True)
    detections = load_detections()
    video_map = {}
    for date in dates:
        path = os.path.join(VIDEO_DIR, date)
        if os.path.isdir(path):
            video_map[date] = sorted(os.listdir(path), reverse=True)
    return render_template("videos.html", video_map=video_map, detections=detections)


@app.route("/videos/<date>/<filename>")
def serve_video(date, filename):
    return send_from_directory(os.path.join(VIDEO_DIR, date), filename)


@app.route("/delete", methods=["POST"])
def delete_video():
    video_rel_path = request.form.get("video_path")
    video_abs_path = os.path.abspath(os.path.join(VIDEO_DIR, video_rel_path))

    if not video_abs_path.startswith(os.path.abspath(VIDEO_DIR)):
        flash("Chemin non autorisé.", "danger")
        return redirect(url_for("index"))

    try:
        if os.path.exists(video_abs_path):
            os.remove(video_abs_path)
            flash(f"Vidéo supprimée : {video_rel_path}", "success")
        else:
            flash("Fichier introuvable.", "warning")
    except Exception as e:
        flash(f"Erreur lors de la suppression : {e}", "danger")

    return redirect(url_for("videos"))


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
