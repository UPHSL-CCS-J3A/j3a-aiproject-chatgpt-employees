import os
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# --- CONFIGURATION (Skin Type) ---

# 🛑 1. UPDATE THIS PATH: Set this to the location of your new skin type image dataset.
# Example: DATASET_ROOT_PATH = r'C:\Users\adria\Desktop\skin_type_dataset'
DATASET_ROOT_PATH = r'C:\Users\adria\Downloads\skin_dataset' 

MODEL_FILE_NAME = "skin_model.h5"
MODEL_SAVE_PATH = os.path.join("models", MODEL_FILE_NAME)
IMAGE_SIZE = (128, 128) # Input size for the CNN
BATCH_SIZE = 32
EPOCHS = 20 # Increased epochs since this is a new training run
NUM_CLASSES = 3 

# Keras reads classes alphabetically by folder name: 0: Dry, 1: Normal, 2: Oily
TARGET_FOLDER_NAMES = ["Dry", "Normal", "Oily"] 

# --- MODEL DEFINITION ---

def build_skin_model(input_shape=(128, 128, 3), num_classes=NUM_CLASSES):
    """Defines a simple CNN architecture tailored for 3 classes."""
    model = Sequential([
        # The CNN architecture remains the same to analyze visual features across the whole image
        Conv2D(32, (3, 3), activation='relu', input_shape=input_shape),
        MaxPooling2D((2, 2)),
        Conv2D(64, (3, 3), activation='relu'),
        MaxPooling2D((2, 2)),
        Conv2D(128, (3, 3), activation='relu'),
        MaxPooling2D((2, 2)),
        Flatten(),
        Dense(512, activation='relu'),
        Dropout(0.5),
        # 🛑 Output layer changed to NUM_CLASSES (3) for Dry, Normal, Oily
        Dense(num_classes, activation='softmax') 
    ])
    model.compile(optimizer='adam',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    return model

def train_and_save_model():
    """Loads data, trains the model, and saves the .h5 file."""
    if not os.path.exists(DATASET_ROOT_PATH):
        print(f"🛑 Error: Dataset path not found at '{DATASET_ROOT_PATH}'. Please check the path and ensure the dataset is ready.")
        return

    # Check that required folders exist
    for folder in TARGET_FOLDER_NAMES:
        full_path = os.path.join(DATASET_ROOT_PATH, folder)
        if not os.path.exists(full_path):
            print(f"🛑 Error: Required class folder not found: {folder}. Ensure you have 'Dry', 'Normal', and 'Oily' subfolders.")
            return

    # --- Data Preparation and Augmentation ---
    datagen = ImageDataGenerator(
        rescale=1./255,
        rotation_range=20,
        validation_split=0.2 
    )
    
    train_generator = datagen.flow_from_directory(
        DATASET_ROOT_PATH,
        classes=TARGET_FOLDER_NAMES,
        target_size=IMAGE_SIZE,
        batch_size=BATCH_SIZE,
        class_mode='categorical',
        subset='training'
    )

    validation_generator = datagen.flow_from_directory(
        DATASET_ROOT_PATH,
        classes=TARGET_FOLDER_NAMES,
        target_size=IMAGE_SIZE,
        batch_size=BATCH_SIZE,
        class_mode='categorical',
        subset='validation'
    )

    if train_generator.num_classes != NUM_CLASSES:
        print(f"FATAL ERROR: Keras found {train_generator.num_classes} classes but expected {NUM_CLASSES}. Check folder names.")
        return
        
    # --- Model Training and Saving ---
    model = build_skin_model()
    print("Starting training...")
    
    model.fit(
        train_generator,
        steps_per_epoch=train_generator.samples // BATCH_SIZE,
        epochs=EPOCHS,
        validation_data=validation_generator,
        validation_steps=validation_generator.samples // BATCH_SIZE
    )

    if not os.path.exists("models"):
        os.makedirs("models")
        
    model.save(MODEL_SAVE_PATH)
    print(f"\n✅ Model successfully saved to: {MODEL_SAVE_PATH}. Ready to run the UI.")

if __name__ == "__main__":
    train_and_save_model()