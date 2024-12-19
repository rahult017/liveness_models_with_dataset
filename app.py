import os
import zipfile
import cv2
import numpy as np
import logging
from datetime import datetime, timedelta
from flask import Flask, jsonify, request, render_template
from werkzeug.utils import secure_filename
from celery import Celery
from pathlib import Path
from io import BytesIO
from tensorflow.keras.models import load_model
from training_model_new import LivenessDetectionTrainer

# Initialize Flask App
app = Flask(__name__)

# Constants
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif', 'bmp', 'tiff', 'zip'}
UPLOAD_FOLDER = os.path.join(os.getcwd(), "uploads")
SAVE_DIRECTORY = Path(UPLOAD_FOLDER)
SAVE_DIRECTORY.mkdir(parents=True, exist_ok=True)
MODEL_SAVE_PATH = "liveness_model.keras"
CLEANUP_DAYS = 30  # Files older than this will be deleted

# Logging Configuration
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# Celery configuration
app.config['CELERY_BROKER_URL'] = 'redis://localhost:6379/0'  # Redis as the broker
app.config['CELERY_RESULT_BACKEND'] = 'redis://localhost:6379/0'
celery = Celery(app.name, broker=app.config['CELERY_BROKER_URL'])
celery.conf.update(app.config)

# Load OpenCV face detection model
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')


def allowed_file(filename):
    """Helper function to check if a file is allowed."""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


@app.route('/')
def index():
    """Render the index page with upload form."""
    logger.info("Rendering index page")
    return render_template('index.html')


@app.route('/upload', methods=['POST'])
def upload_files():
    """
    Handles uploading a single image or a ZIP file and provides liveness detection results.
    Logs operations and calculates results for individual and overall files.
    """
    if 'file' not in request.files:
        logger.error("No file provided in request")
        return jsonify({"error": "No file provided"}), 400

    file = request.files['file']
    if file.filename == '' or not allowed_file(file.filename):
        logger.error("Invalid file format or empty file name")
        return jsonify({"error": "Invalid file format"}), 400

    logger.info(f"Processing file: {file.filename}")
    results = []
    real_count, fake_count = 0, 0

    trainer = LivenessDetectionTrainer(
        model_save_path=MODEL_SAVE_PATH
        image_input=file_path,
        max_workers=4  # Adjust based on CPU cores
    )
    trainer.prepare_data()
    trainer.build_or_load_model()
    history = trainer.train_model()

    if file.filename.endswith('.zip'):
        logger.info("Detected ZIP file. Extracting and processing images.")
        zip_file = zipfile.ZipFile(BytesIO(file.read()))
        for filename in zip_file.namelist():
            if allowed_file(filename):
                img_data = zip_file.read(filename)
                file_path = os.path.join(SAVE_DIRECTORY, secure_filename(filename))
                with open(file_path, 'wb') as f:
                    f.write(img_data)
                result = process_and_get_response(file_path, trainer)
                results.append(result)
                if result.get("label") == "Real":
                    real_count += 1
                elif result.get("label") == "Fake":
                    fake_count += 1
    else:
        logger.info("Detected single file upload. Processing image.")
        filename = secure_filename(file.filename)
        file_path = os.path.join(SAVE_DIRECTORY, filename)
        file.save(file_path)
        result = process_and_get_response(file_path, trainer)
        results.append(result)
        if result.get("label") == "Real":
            real_count += 1
        elif result.get("label") == "Fake":
            fake_count += 1

    overall_summary = {
        "total_files": len(results),
        "real_count": real_count,
        "fake_count": fake_count
    }

    logger.info(f"Overall Summary: {overall_summary}")
    return jsonify({"results": results, "summary": overall_summary})


def process_and_get_response(file_path, trainer):
    """
    Processes a single image file for liveness detection and constructs a response.
    """
    try:
        face_detected, error_message, face_data = detect_face(file_path)
        if face_detected:
            result = trainer.predict_single_image(file_path)
            face_results = [{"x": int(face["x"]), "y": int(face["y"]), 
                             "width": int(face["width"]), "height": int(face["height"])} for face in face_data]
            response = {
                "file_name": os.path.basename(file_path),
                "face_result": face_results,
                "label": result["label"],
                "confidence": float(result["confidence"]),
                "accuracy": float(result["accuracy"]),
                "val_loss": float(result["val_loss"]),
                "learning_rate": float(result["learning_rate"]),
                "loss": float(result["loss"]),
                "val_accuracy": float(result["val_accuracy"])
            }
        else:
            response = {
                "file_name": os.path.basename(file_path),
                "error": error_message,
                "label": "Unknown",
                "confidence": None,
                "face_result": []
            }
        return response
    except Exception as e:
        logger.error(f"Error processing file {file_path}: {str(e)}")
        return {"file_name": os.path.basename(file_path), "error": str(e)}


def detect_face(image_path):
    """
    Detect faces in the image using OpenCV.
    """
    image = cv2.imread(image_path)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    if len(faces) == 0:
        logger.warning(f"No face detected in {image_path}")
        return False, "No face detected in the image", None

    logger.info(f"Detected {len(faces)} face(s) in {image_path}")
    face_data = [{"x": x, "y": y, "width": w, "height": h} for (x, y, w, h) in faces]
    return True, None, face_data





@celery.task
def clean_old_files():
    """Clean files older than a specified number of days."""
    logger.info("Starting cleanup of old files.")
    cutoff_date = datetime.now() - timedelta(days=CLEANUP_DAYS)
    for file in SAVE_DIRECTORY.iterdir():
        if file.is_file() and datetime.fromtimestamp(file.stat().st_mtime) < cutoff_date:
            logger.info(f"Deleting old file: {file}")
            file.unlink()
    logger.info("Old files deleted successfully.")
    return "Old files deleted successfully"


if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)
