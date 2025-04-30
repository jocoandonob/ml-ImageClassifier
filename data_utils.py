import numpy as np
import tensorflow as tf
import os
import zipfile
import tempfile
import shutil
from PIL import Image
import random
from sklearn.model_selection import train_test_split

def prepare_dataset(images, labels, augment=False, batch_size=32):
    """
    Create a TensorFlow dataset from numpy arrays.
    
    Args:
        images: Numpy array of images
        labels: Numpy array of labels
        augment: Whether to use data augmentation
        batch_size: Batch size for the dataset
        
    Returns:
        TensorFlow dataset
    """
    dataset = tf.data.Dataset.from_tensor_slices((images, labels))
    
    if augment:
        # Basic data augmentation
        data_augmentation = tf.keras.Sequential([
            tf.keras.layers.RandomFlip("horizontal"),
            tf.keras.layers.RandomRotation(0.1),
            tf.keras.layers.RandomZoom(0.1),
            tf.keras.layers.RandomTranslation(0.1, 0.1),
            tf.keras.layers.RandomBrightness(0.1)
        ])
        
        def augment_map_fn(image, label):
            image = data_augmentation(image)
            return image, label
        
        dataset = dataset.map(augment_map_fn, num_parallel_calls=tf.data.AUTOTUNE)
    
    dataset = dataset.cache().shuffle(buffer_size=1000).batch(batch_size).prefetch(tf.data.AUTOTUNE)
    return dataset

def load_and_preprocess_data(zip_path, target_size=(32, 32), batch_size=32, validation_split=0.2, test_split=0.1):
    """
    Load and preprocess a custom dataset from a zip file.
    
    Args:
        zip_path: Path to the zip file containing the dataset
        target_size: Target size for the images
        batch_size: Batch size for the dataset
        validation_split: Proportion of data to use for validation
        test_split: Proportion of data to use for testing
        
    Returns:
        train_dataset, valid_dataset, test_dataset, class_names, test_images, test_labels
    """
    # Create a temporary directory to extract the dataset
    with tempfile.TemporaryDirectory() as temp_dir:
        # Extract the zip file
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(temp_dir)
        
        # Find all class directories
        class_dirs = []
        for root, dirs, files in os.walk(temp_dir):
            for dir_name in dirs:
                if not dir_name.startswith('.'):  # Skip hidden directories
                    class_path = os.path.join(root, dir_name)
                    if any([f.lower().endswith(('.jpg', '.jpeg', '.png')) for f in os.listdir(class_path)]):
                        class_dirs.append(class_path)
        
        if not class_dirs:
            raise ValueError("No valid class directories found in the dataset.")
        
        # Get class names from directory names
        class_names = [os.path.basename(dir_path) for dir_path in class_dirs]
        
        # Load images and labels
        images = []
        labels = []
        
        for class_idx, class_dir in enumerate(class_dirs):
            # Get all image files in the class directory
            image_files = [f for f in os.listdir(class_dir) 
                          if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
            
            for image_file in image_files:
                try:
                    img_path = os.path.join(class_dir, image_file)
                    img = Image.open(img_path)
                    img = img.convert('RGB')  # Convert to RGB
                    img = img.resize(target_size)  # Resize
                    img_array = np.array(img) / 255.0  # Normalize to [0, 1]
                    
                    images.append(img_array)
                    labels.append(class_idx)
                except Exception as e:
                    print(f"Error processing {img_path}: {str(e)}")
        
        if not images:
            raise ValueError("No valid images found in the dataset.")
        
        # Convert to numpy arrays
        images = np.array(images, dtype=np.float32)
        labels = np.array(labels)
        
        # One-hot encode the labels
        num_classes = len(class_names)
        labels_one_hot = tf.keras.utils.to_categorical(labels, num_classes)
        
        # Split the data into train, validation, and test sets
        X_train, X_temp, y_train, y_temp = train_test_split(
            images, labels_one_hot, test_size=validation_split + test_split, random_state=42
        )
        
        X_valid, X_test, y_valid, y_test = train_test_split(
            X_temp, y_temp, test_size=test_split / (validation_split + test_split), random_state=42
        )
        
        # Create datasets
        train_dataset = prepare_dataset(X_train, y_train, augment=True, batch_size=batch_size)
        valid_dataset = prepare_dataset(X_valid, y_valid, augment=False, batch_size=batch_size)
        test_dataset = prepare_dataset(X_test, y_test, augment=False, batch_size=batch_size)
        
        return train_dataset, valid_dataset, test_dataset, class_names, X_test, y_test
