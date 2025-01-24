import cv2
import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator

def preprocess_image(image_path, target_size=(48, 48)):
    """
    Load and preprocess an image for emotion detection.
    """
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    image = cv2.resize(image, target_size)
    image = image / 255.0  # Normalize to [0, 1]
    return np.expand_dims(image, axis=-1)

def augment_data(images, labels):
    """
    Perform data augmentation using ImageDataGenerator.
    """
    datagen = ImageDataGenerator(
        rotation_range=10,
        width_shift_range=0.1,
        height_shift_range=0.1,
        shear_range=0.1,
        zoom_range=0.1,
        horizontal_flip=True
    )
    return datagen.flow(images, labels, batch_size=32)
