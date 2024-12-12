import os
import pandas as pd
from tensorflow.keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array
from tensorflow.keras.models import load_model
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from build_model_training import build_model
import numpy as np
import zipfile
from pathlib import Path
import json
import stat
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
from functools import partial, lru_cache
import shutil
from typing import Union, List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, field
import logging
from contextlib import contextmanager
import tempfile
import asyncio
from aiofiles import open as aio_open
from PIL import Image
import io

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

@dataclass
class ModelMetrics:
    accuracy: float = 0.0
    val_accuracy: float = 0.0 
    loss: float = 0.0
    val_loss: float = 0.0
    success_rate: float = 0.0
    rejection_rate: float = 0.0
    total_predictions: int = 0
    successful_predictions: int = 0
    rejected_predictions: int = 0
    latest_prediction: Dict[str, Any] = field(default_factory=dict)

class LivenessDetectionTrainer:
    def __init__(self, image_input: Union[str, Path], label_csv_path: str, model_save_path: str,
                 image_size: Tuple[int, int] = (224, 224), batch_size: int = 32, epochs: int = 50,
                 max_workers: int = None, confidence_threshold: float = 0.8):
        """
        Enhanced trainer with dynamic resource allocation and advanced error handling
        Args:
            image_input: Path to image directory or zip file
            label_csv_path: Path to save/load labels CSV
            model_save_path: Path to save/load model
            image_size: Target size for images
            batch_size: Batch size for training
            epochs: Number of training epochs
            max_workers: Maximum number of workers for parallel processing (None for auto)
            confidence_threshold: Threshold for prediction confidence
        """
        self.image_input = Path(image_input)
        self.max_workers = max_workers or os.cpu_count()
        self.valid_extensions = {'.png', '.jpg', '.jpeg', '.gif', '.bmp', '.tiff'}
        self.image_dir = self._setup_image_input()
        self.label_csv_path = Path(label_csv_path)
        self.model_save_path = Path(model_save_path)
        self.image_size = image_size
        self.batch_size = batch_size
        self.epochs = epochs
        self.confidence_threshold = confidence_threshold
        self.train_generator = None
        self.val_generator = None
        self.model = None
        self.metrics = ModelMetrics()
        self._setup_directories()
        
        # Setup memory-efficient caching
        self._image_cache = lru_cache(maxsize=1000)(self._load_and_preprocess_image)

    @contextmanager
    def _error_handler(self, operation: str):
        """Context manager for unified error handling"""
        try:
            yield
        except PermissionError as e:
            logger.error(f"Permission error during {operation}: {e}")
            raise
        except Exception as e:
            logger.error(f"Error during {operation}: {e}")
            raise

    async def _async_file_operation(self, path: Path, mode: str, operation: callable):
        """Asynchronous file operation wrapper"""
        async with aio_open(str(path), mode) as f:
            return await operation(f)

    def _setup_directories(self) -> None:
        """Create necessary directories with proper permissions and logging"""
        for path in [self.image_dir, self.label_csv_path.parent, self.model_save_path.parent]:
            with self._error_handler("directory setup"):
                path.mkdir(parents=True, exist_ok=True)
                os.chmod(str(path), stat.S_IWRITE | stat.S_IREAD)
                logger.info(f"Directory setup completed: {path}")

    def _setup_image_input(self) -> Path:
        """Enhanced image input handling with progress tracking"""
        if self.image_input.is_file() and self.image_input.suffix.lower() == '.zip':
            extract_path = Path(tempfile.mkdtemp()) / 'extracted_images'
            with self._error_handler("zip extraction"):
                with zipfile.ZipFile(str(self.image_input)) as zip_ref:
                    total_files = len(zip_ref.namelist())
                    for i, member in enumerate(zip_ref.namelist(), 1):
                        zip_ref.extract(member, str(extract_path))
                        if i % 100 == 0:
                            logger.info(f"Extracted {i}/{total_files} files")
                logger.info(f"Extraction completed to {extract_path}")
            return extract_path
        return self.image_input

    async def _process_image_batch_async(self, files: List[Path]) -> List[Dict[str, Any]]:
        """Asynchronous batch image processing"""
        results = []
        for file in files:
            if file.suffix.lower() in self.valid_extensions:
                try:
                    with Image.open(file) as img:
                        # Verify image integrity
                        img.verify()
                        rel_path = os.path.relpath(file, start=self.image_dir)
                        label = 0 if 'fake' in str(file).lower() else 1
                        results.append({'filename': rel_path, 'label': label})
                except Exception as e:
                    logger.warning(f"Skipping corrupted image {file}: {e}")
        return results

    def generate_labels_dynamic(self) -> pd.DataFrame:
        """Generate labels using hybrid parallel processing"""
        image_files = list(Path(self.image_dir).rglob('*'))
        batch_size = max(100, len(image_files) // self.max_workers)
        batches = [image_files[i:i + batch_size] for i in range(0, len(image_files), batch_size)]

        async def process_all_batches():
            tasks = [self._process_image_batch_async(batch) for batch in batches]
            return await asyncio.gather(*tasks)

        results = asyncio.run(process_all_batches())
        data = [item for batch in results for item in batch]
        
        if not data:
            raise ValueError("No valid image files found")
            
        df = pd.DataFrame(data)
        df.to_csv(self.label_csv_path, index=False)
        logger.info(f"Labels CSV created with {len(df)} entries at: {self.label_csv_path}")
        return df

    def prepare_data(self) -> None:
        """Enhanced data preparation with validation and augmentation"""
        with self._error_handler("data preparation"):
            df = pd.read_csv(self.label_csv_path) if self.label_csv_path.exists() else self.generate_labels_dynamic()
            
            # Validate dataset
            df['label'] = df['label'].astype(str)
            class_distribution = df['label'].value_counts()
            logger.info(f"Class distribution: {class_distribution.to_dict()}")
            
            if df['label'].nunique() < 2:
                raise ValueError("Dataset must contain at least two classes")

            # Advanced data augmentation
            datagen = ImageDataGenerator(
                rescale=1.0/255,
                validation_split=0.2,
                rotation_range=20,
                width_shift_range=0.2,
                height_shift_range=0.2,
                horizontal_flip=True,
                zoom_range=0.2,
                shear_range=0.2,
                fill_mode='nearest',
                brightness_range=[0.8, 1.2]
            )

            self.train_generator = self._create_generator(datagen, df, 'training')
            self.val_generator = self._create_generator(datagen, df, 'validation')
            logger.info(f"Data generators created. Class indices: {self.train_generator.class_indices}")

    def _create_generator(self, datagen: ImageDataGenerator, df: pd.DataFrame, subset: str) -> Any:
        """Create optimized data generator with error handling"""
        return datagen.flow_from_dataframe(
            dataframe=df,
            directory=self.image_dir,
            x_col='filename',
            y_col='label',
            target_size=self.image_size,
            batch_size=self.batch_size,
            class_mode='binary',
            subset=subset,
            shuffle=subset == 'training'
        )

    def build_or_load_model(self) -> None:
        """Enhanced model loading with validation"""
        with self._error_handler("model loading"):
            if self.model_save_path.exists():
                self.model = load_model(self.model_save_path)
                logger.info("Loaded existing model")
                # Validate model architecture
                self.model.summary()
            else:
                self.model = build_model()
                logger.info("Built new model")

    def train_model(self) -> dict:
        """Enhanced training with advanced callbacks and monitoring"""
        callbacks = [
            EarlyStopping(
                monitor='val_loss',
                patience=5,
                restore_best_weights=True,
                verbose=1
            ),
            ReduceLROnPlateau(
                monitor='val_loss',
                patience=3,
                verbose=1,
                factor=0.5,
                min_lr=1e-6,
                cooldown=2
            ),
            ModelCheckpoint(
                self.model_save_path,
                monitor='val_loss',
                save_best_only=True,
                verbose=1,
                save_weights_only=False
            )
        ]

        self.model.compile(
            optimizer='adam',
            loss='binary_crossentropy',
            metrics=['accuracy', 'AUC', 'Precision', 'Recall']
        )

        history = self.model.fit(
            self.train_generator,
            epochs=self.epochs,
            validation_data=self.val_generator,
            callbacks=callbacks,
            verbose=1
        )

        self._save_model_and_metrics(history)
        return history

    async def _save_model_and_metrics_async(self, metrics: Dict[str, Any]) -> None:
        """Asynchronous metrics saving"""
        async with aio_open('metrics.json', 'w') as f:
            await f.write(json.dumps(metrics, indent=4))

    def _save_model_and_metrics(self, history: dict) -> None:
        """Enhanced model and metrics saving with validation"""
        with self._error_handler("saving model and metrics"):
            self.model.save(self.model_save_path)
            
            # Update metrics
            self.metrics.accuracy = float(history.history['accuracy'][-1])
            self.metrics.val_accuracy = float(history.history['val_accuracy'][-1])
            self.metrics.loss = float(history.history['loss'][-1])
            self.metrics.val_loss = float(history.history['val_loss'][-1])
            
            metrics_dict = {k: v for k, v in self.metrics.__dict__.items()}
            asyncio.run(self._save_model_and_metrics_async(metrics_dict))
            logger.info("Model and metrics saved successfully")

    async def predict_single_image_async(self, image_path: Union[str, Path]) -> Tuple[str, float]:
        """Asynchronous single image prediction"""
        image_path = Path(image_path)
        if not image_path.exists():
            raise FileNotFoundError(f"Image not found: {image_path}")

        img = await asyncio.to_thread(self._image_cache, image_path)
        prediction = await asyncio.to_thread(self.model.predict, img, verbose=0)
        prediction = prediction[0]
        
        result = "Real" if prediction > 0.5 else "Fake"
        confidence = prediction if prediction > 0.5 else 1 - prediction
        
        await self._update_metrics_async(result, confidence)
        return result, float(confidence)

    def predict_single_image(self, image_path: Union[str, Path]) -> Tuple[str, float]:
        """Synchronous wrapper for async prediction"""
        return asyncio.run(self.predict_single_image_async(image_path))

    @lru_cache(maxsize=1000)
    def _load_and_preprocess_image(self, image_path: Path) -> np.ndarray:
        """Cached image loading and preprocessing"""
        with self._error_handler("image loading"):
            img = load_img(str(image_path), target_size=self.image_size)
            return np.expand_dims(img_to_array(img), axis=0) / 255.0

    async def _update_metrics_async(self, result: str, confidence: float) -> None:
        """Asynchronous metrics update"""
        if confidence >= self.confidence_threshold:
            self.metrics.successful_predictions += 1
        else:
            self.metrics.rejected_predictions += 1
        
        self.metrics.total_predictions += 1
        self._update_rates()
        self.metrics.latest_prediction = {
            'result': result,
            'confidence': float(confidence)
        }
        
        await self._save_model_and_metrics_async(self.metrics.__dict__)

    def _update_rates(self) -> None:
        """Update success and rejection rates"""
        total = self.metrics.total_predictions
        if total > 0:
            self.metrics.success_rate = self.metrics.successful_predictions / total
            self.metrics.rejection_rate = self.metrics.rejected_predictions / total
            

    async def predict_multiple_images_async(self, image_paths: Union[str, List[str], Path]) -> List[Dict[str, Any]]:
        """Asynchronous batch prediction"""
        if isinstance(image_paths, (str, Path)):
            image_paths = [image_paths]
            
        if any(str(path).endswith('.zip') for path in image_paths):
            image_paths = await asyncio.to_thread(self._extract_from_zip, image_paths[0])

        tasks = [self.predict_single_image_async(path) for path in image_paths]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        return [
            {
                'image': str(path),
                'prediction': result[0],
                'confidence': result[1]
            } if not isinstance(result, Exception) else {
                'image': str(path),
                'error': str(result)
            }
            for path, result in zip(image_paths, results)
        ]

    def predict_multiple_images(self, image_paths: Union[str, List[str], Path]) -> List[Dict[str, Any]]:
        """Synchronous wrapper for async batch prediction"""
        return asyncio.run(self.predict_multiple_images_async(image_paths))

    def _extract_from_zip(self, zip_path: Union[str, Path]) -> List[Path]:
        """Extract images from zip efficiently"""
        temp_dir = Path(tempfile.mkdtemp())
        
        with zipfile.ZipFile(str(zip_path)) as zip_ref:
            zip_ref.extractall(str(temp_dir))
            
        image_paths = [f for f in temp_dir.rglob('*') 
                      if f.suffix.lower() in self.valid_extensions]
        
        return image_paths

    def __del__(self):
        """Cleanup temporary files on object destruction"""
        try:
            for path in Path(tempfile.gettempdir()).glob('tmp*'):
                if path.is_dir() and path.exists():
                    shutil.rmtree(path)
        except Exception as e:
            logger.warning(f"Error during cleanup: {e}")


if __name__ == "__main__":
    # Example usage
    IMAGE_INPUT = 'D:/liveness_detection/Dataset'
    LABEL_CSV_PATH = 'labels.csv'
    MODEL_SAVE_PATH = "liveness_model.keras"

    trainer = LivenessDetectionTrainer(
        image_input=IMAGE_INPUT,
        label_csv_path=LABEL_CSV_PATH,
        model_save_path=MODEL_SAVE_PATH,
        max_workers=4  # Adjust based on CPU cores
    )

    # Training workflow
    trainer.prepare_data()
    trainer.build_or_load_model()
    history = trainer.train_model()

    # Example predictions
    # result, confidence = trainer.predict_single_image("path/to/test_image.jpg")
    # print(f"Single image prediction: {result} (Confidence: {confidence:.2f})")

    # Batch predictions
    # results = trainer.predict_multiple_images("path/to/test_images.zip")
    # for result in results:
    #     if 'error' in result:
    #         print(f"Error processing {result['image']}: {result['error']}")
    #     else:
    #         print(f"Image: {result['image']}")
    #         print(f"Prediction: {result['prediction']}")
    #         print(f"Confidence: {result['confidence']:.2f}")
    #     print("---")
