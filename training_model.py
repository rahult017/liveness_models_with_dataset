import os
import json
import shutil
import zipfile
import asyncio
import logging
import tempfile
import numpy as np
from pathlib import Path
from PIL import Image
from functools import lru_cache
from typing import Union, Dict, Any, Tuple
from dataclasses import dataclass, field
from tensorflow.keras.models import load_model
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from build_model_training import build_model  # Replace with actual model builder

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

@dataclass
class ModelMetrics:
    accuracy: float = 0.0
    val_accuracy: float = 0.0
    loss: float = 0.0
    val_loss: float = 0.0
    latest_prediction: Dict[str, Any] = field(default_factory=dict)
    training_history: Dict[str, list] = field(default_factory=lambda: {'accuracy': [], 'val_accuracy': [], 'loss': [], 'val_loss': []})

    # Metrics for predictions
    total_predictions: int = 0
    successful_predictions: int = 0
    rejected_predictions: int = 0
    confidence: float = 0.0
    success_rate: float = 0.0
    rejection_rate: float = 0.0

    def update_from_history(self, history):
        """Update metrics from training history"""
        if history and history.history:
            self.accuracy = float(history.history['accuracy'][-1])
            self.val_accuracy = float(history.history['val_accuracy'][-1])
            self.loss = float(history.history['loss'][-1])
            self.val_loss = float(history.history['val_loss'][-1])
            self.training_history['accuracy'].extend(history.history['accuracy'])
            self.training_history['val_accuracy'].extend(history.history['val_accuracy'])
            self.training_history['loss'].extend(history.history['loss'])
            self.training_history['val_loss'].extend(history.history['val_loss'])
            logger.info(f"Updated training metrics - Accuracy: {self.accuracy:.4f}, Val Accuracy: {self.val_accuracy:.4f}")

    def update_prediction_metrics(self, confidence: float, threshold: float = 0.8):
        """Update prediction-related metrics"""
        self.total_predictions += 1
        self.confidence = float(confidence)
        if confidence >= threshold:
            self.successful_predictions += 1
        else:
            self.rejected_predictions += 1
        self._calculate_rates()
        logger.info(f"Prediction Metrics Updated - Success Rate: {self.success_rate:.4f}, Rejection Rate: {self.rejection_rate:.4f}")

    def _calculate_rates(self):
        """Calculate success and rejection rates"""
        if self.total_predictions > 0:
            self.success_rate = self.successful_predictions / self.total_predictions
            self.rejection_rate = self.rejected_predictions / self.total_predictions

class LivenessDetectionTrainer:
    def __init__(self, image_input: Union[str, Path], label_csv_path: str, model_save_path: str,
                 image_size: Tuple[int, int] = (224, 224), batch_size: int = 32, epochs: int = 50,
                 max_workers: int = None, confidence_threshold: float = 0.8):
        self.image_input = Path(image_input)
        self.label_csv_path = Path(label_csv_path)
        self.model_save_path = Path(model_save_path)
        self.image_size = image_size
        self.batch_size = batch_size
        self.epochs = epochs
        self.confidence_threshold = confidence_threshold
        self.metrics = ModelMetrics()

        self.image_dir = Path(tempfile.mkdtemp())
        self._extract_images()
        self.train_generator = None
        self.val_generator = None
        self.model = None

        self._image_cache = lru_cache(maxsize=1000)(self._load_and_preprocess_image)

    def _extract_images(self):
        """Extract images from a zip file if provided"""
        if str(self.image_input).endswith('.zip'):
            with zipfile.ZipFile(self.image_input, 'r') as zip_ref:
                zip_ref.extractall(self.image_dir)
            logger.info(f"Extracted images to {self.image_dir}")
        else:
            shutil.copytree(str(self.image_input), str(self.image_dir), dirs_exist_ok=True)

    def _load_and_preprocess_image(self, image_path: Path) -> np.ndarray:
        """Load and preprocess a single image"""
        img = Image.open(image_path).resize(self.image_size)
        img_array = np.array(img) / 255.0
        return np.expand_dims(img_array, axis=0)

    def build_or_load_model(self):
        """Build or load model"""
        if self.model_save_path.exists():
            logger.info(f"Loading model from {self.model_save_path}")
            self.model = load_model(self.model_save_path)
        else:
            logger.info("Building a new model")
            self.model = build_model(input_shape=(*self.image_size, 3))

    def prepare_data(self):
        """Prepare data generators"""
        datagen = ImageDataGenerator(
            rescale=1./255,
            rotation_range=20,
            width_shift_range=0.2,
            height_shift_range=0.2,
            shear_range=0.2,
            zoom_range=0.2,
            horizontal_flip=True,
            validation_split=0.2
        )
        self.train_generator = datagen.flow_from_directory(
            self.image_dir,
            target_size=self.image_size,
            batch_size=self.batch_size,
            class_mode='binary',
            subset='training'
        )
        self.val_generator = datagen.flow_from_directory(
            self.image_dir,
            target_size=self.image_size,
            batch_size=self.batch_size,
            class_mode='binary',
            subset='validation'
        )

    def train_model(self):
        """Train model and update metrics"""
        callbacks = [
            EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True),
            ReduceLROnPlateau(monitor='val_loss', patience=3, factor=0.5, min_lr=1e-6),
            ModelCheckpoint(self.model_save_path, monitor='val_loss', save_best_only=True)
        ]
        self.model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
        history = self.model.fit(
            self.train_generator,
            epochs=self.epochs,
            validation_data=self.val_generator,
            callbacks=callbacks
        )
        self.metrics.update_from_history(history)
        self._save_model_and_metrics()

    def _save_model_and_metrics(self):
        """Save model and metrics"""
        try:
            self.model.save(self.model_save_path)
            metrics_dict = {
                'accuracy': self.metrics.accuracy,
                'val_accuracy': self.metrics.val_accuracy,
                'loss': self.metrics.loss,
                'val_loss': self.metrics.val_loss,
                'success_rate': self.metrics.success_rate,
                'rejection_rate': self.metrics.rejection_rate,
                'total_predictions': self.metrics.total_predictions,
                'successful_predictions': self.metrics.successful_predictions,
                'rejected_predictions': self.metrics.rejected_predictions,
                'confidence': self.metrics.confidence,
                'training_history': self.metrics.training_history
            }
            with open('metrics.json', 'w') as f:
                json.dump(metrics_dict, f, indent=4)
        except Exception as e:
            logger.error(f"Error saving model or metrics: {e}")

    async def predict_single_image_async(self, image_path: Union[str, Path]) -> Tuple[str, float]:
        """Async prediction"""
        image_path = Path(image_path)
        if not image_path.exists():
            raise FileNotFoundError(f"Image not found: {image_path}")
        img = await asyncio.to_thread(self._image_cache, image_path)
        prediction = self.model.predict(img)[0]
        result = "Real" if prediction > 0.5 else "Fake"
        confidence = float(prediction if prediction > 0.5 else 1 - prediction)
        self.metrics.update_prediction_metrics(confidence, self.confidence_threshold)
        return result, confidence

    def predict_single_image(self, image_path: Union[str, Path]) -> Tuple[str, float]:
        """Sync prediction"""
        return asyncio.run(self.predict_single_image_async(image_path))

if __name__ == "__main__":
    trainer = LivenessDetectionTrainer(
        image_input='D:/liveness_detection/Dataset',
        label_csv_path='labels.csv',
        model_save_path="liveness_model.keras"
    )
    trainer.prepare_data()
    trainer.build_or_load_model()
    trainer.train_model()
    print(f"Final Metrics: {trainer.metrics.__dict__}")
