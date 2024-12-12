import cv2
from tensorflow.keras.preprocessing.image import img_to_array
import numpy as np

def preprocess_image(
    image_path,
    cascade_path=cv2.data.haarcascades + 'haarcascade_frontalface_default.xml',
    target_size=(224, 224),
    scale_factor=1.1,
    min_neighbors=5,
    min_size=(30, 30),
    return_all_faces=False,
    use_dnn=False,
    dnn_model_path="deploy.prototxt",
    dnn_weights_path="res10_300x300_ssd_iter_140000.caffemodel",
    confidence_threshold=0.5,
    dynamic_threshold=False,
    min_confidence=0.4,
    max_confidence=0.9
):
    """
    Preprocess an image for a face detection model with improved accuracy.
    
    Args:
        image_path (str): Path to the input image.
        cascade_path (str): Path to the Haar cascade XML file for face detection.
        target_size (tuple): Target size to resize the detected face, e.g., (224, 224).
        scale_factor (float): Scale factor for the Haar cascade face detection algorithm.
        min_neighbors (int): Minimum number of neighbors for a Haar cascade detection.
        min_size (tuple): Minimum size of detected faces.
        return_all_faces (bool): If True, returns all detected faces; otherwise, returns the first face.
        use_dnn (bool): If True, uses DNN-based face detection.
        dnn_model_path (str): Path to the DNN model's deploy file (used if use_dnn is True).
        dnn_weights_path (str): Path to the DNN model's weights file (used if use_dnn is True).
        confidence_threshold (float): Confidence threshold for DNN-based face detection.
        dynamic_threshold (bool): If True, dynamically adjusts the confidence threshold based on image content.
        min_confidence (float): Minimum confidence value for face detection.
        max_confidence (float): Maximum confidence value for face detection.

    Returns:
        np.ndarray or None: Preprocessed face(s) as a numpy array or None if no face is detected.
    """
    try:
        # Load the image
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError("Image could not be loaded. Check the image path.")

        # Prepare the image
        (h, w) = image.shape[:2]
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Choose face detection method
        faces = []
        if use_dnn:
            # DNN-based face detection
            net = cv2.dnn.readNetFromCaffe(dnn_model_path, dnn_weights_path)
            blob = cv2.dnn.blobFromImage(image, scalefactor=1.0, size=(300, 300), mean=(104.0, 177.0, 123.0))
            net.setInput(blob)
            detections = net.forward()

            # Adjust confidence threshold dynamically
            if dynamic_threshold:
                dynamic_confidence = min_confidence + (max_confidence - min_confidence) * (h / 1000)
                dynamic_confidence = min(max(dynamic_confidence, min_confidence), max_confidence)
                print(f"Dynamic threshold applied: {dynamic_confidence}")
                confidence_threshold = dynamic_confidence

            for i in range(0, detections.shape[2]):
                confidence = detections[0, 0, i, 2]
                if confidence > confidence_threshold:
                    box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                    (x, y, x1, y1) = box.astype("int")
                    faces.append((x, y, x1 - x, y1 - y))

        else:
            # Haar cascade-based face detection
            face_cascade = cv2.CascadeClassifier(cascade_path)
            if face_cascade.empty():
                raise ValueError("Haar cascade file could not be loaded. Check the cascade path.")

            faces = face_cascade.detectMultiScale(
                gray,
                scaleFactor=scale_factor,
                minNeighbors=min_neighbors,
                minSize=min_size
            )

        if len(faces) == 0:
            return None  # No faces detected

        # Process detected faces
        processed_faces = []
        for (x, y, w, h) in faces:
            face = image[y:y+h, x:x+w]
            face = cv2.resize(face, target_size)  # Resize to match the model input
            face = face.astype("float32") / 255.0
            face = img_to_array(face)
            processed_faces.append(np.expand_dims(face, axis=0))

        return processed_faces if return_all_faces else processed_faces[0]

    except Exception as e:
        print(f"Error during preprocessing: {e}")
        return None


