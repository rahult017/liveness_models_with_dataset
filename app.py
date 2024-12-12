import os
import zipfile
import json
import pandas as pd
import numpy as np
import cv2
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import load_model
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from flask import Flask, jsonify, request, render_template
from werkzeug.utils import secure_filename
from io import BytesIO
from build_model_training import build_model
from pathlib import Path
from training_model import LivenessDetectionTrainer

app = Flask(__name__)

# Constants
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif', 'bmp', 'tiff'}
UPLOAD_FOLDER = os.path.join(os.getcwd(), "uploads")
SAVE_DIRECTORY = Path(UPLOAD_FOLDER)
SAVE_DIRECTORY.mkdir(parents=True, exist_ok=True)
MODEL_SAVE_PATH = "liveness_model.keras"
METRICS_LOG_FILE = "metrics.json"

# Helper function to check if a file is allowed
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS



# Flask Routes
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload_and_train', methods=['POST'])
def upload_and_train():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400

    file = request.files['file']

    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400

    # Handle ZIP file
    if file.filename.endswith('.zip'):
        zip_file_path = os.path.join(UPLOAD_FOLDER, secure_filename(file.filename))
        file.save(zip_file_path)

        # Extract ZIP file contents
        with zipfile.ZipFile(zip_file_path, 'r') as zip_ref:
            zip_ref.extractall(UPLOAD_FOLDER)

    # Handle single image file
    elif allowed_file(file.filename):
        filename = secure_filename(file.filename)
        file_path = os.path.join(UPLOAD_FOLDER, filename)
        file.save(file_path)

    else:
        return jsonify({'error': 'Unsupported file type'}), 400

    # Start training process
    try:
        trainer = LivenessDetectionTrainer(
            image_dir=UPLOAD_FOLDER, 
            label_csv_path='labels.csv', 
            model_save_path=MODEL_SAVE_PATH
        )
        trainer.prepare_data()
        trainer.build_or_load_model()
        history = trainer.train_model()

        latest_metrics = {
            "accuracy": history.history.get("accuracy", [])[-1],
            "val_accuracy": history.history.get("val_accuracy", [])[-1],
            "loss": history.history.get("loss", [])[-1],
            "val_loss": history.history.get("val_loss", [])[-1]
        }

        return jsonify({
            "message": "Model trained successfully!",
            "metrics": latest_metrics
        }), 200

    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/analyze', methods=['GET'])
def analyze():
    if not os.path.exists(METRICS_LOG_FILE):
        return jsonify({"error": "Metrics log not found. Train the model first."}), 404

    try:
        with open(METRICS_LOG_FILE, "r") as f:
            data = json.load(f)

        latest_metrics = {
            "accuracy": data.get("accuracy", [])[-1] if data.get("accuracy") else 0,
            "val_accuracy": data.get("val_accuracy", [])[-1] if data.get("val_accuracy") else 0,
            "loss": data.get("loss", [])[-1] if data.get("loss") else 0,
            "val_loss": data.get("val_loss", [])[-1] if data.get("val_loss") else 0,
            "learning_rate": data.get("learning_rate", [])[0] if data.get("learning_rate") else 0
        }

        return jsonify(latest_metrics)

    except Exception as e:
        return jsonify({"error": f"Failed to read metrics: {str(e)}"}), 500


if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)
