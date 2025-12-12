import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import numpy as np
import matplotlib.pyplot as plt
import os
import shutil
from pathlib import Path
import json
from PIL import Image

# Configuration
IMG_SIZE = 150
BATCH_SIZE = 32
EPOCHS = 25

# Note: You'll need to download the dataset from Kaggle first
# [https://www.kaggle.com/datasets/hizkiasiregar/smart-wardrobe-clothing-dataset](https://www.kaggle.com/datasets/hizkiasiregar/smart-wardrobe-clothing-dataset)
# Extract it and update these paths accordingly
TRAIN_DIR = 'Hizkia Siregar - Homework ML'  # Original folder with clothing images in category subfolders
ORGANIZED_DIR = 'clothing_organized'  # Will be created if needed
TEST_DIR = 'test'  # Update with your path

MODEL_PATH = 'clothing_classifier_model.h5'
CLASS_NAMES_PATH = 'class_names.json'

def check_model_exists():
    """
    Check if trained model already exists
    """
    return os.path.exists(MODEL_PATH) and os.path.exists(CLASS_NAMES_PATH)

def load_saved_model():
    """
    Load saved model and class names
    """
    try:
        model = keras.models.load_model(MODEL_PATH)
        with open(CLASS_NAMES_PATH, 'r') as f:
            class_names = json.load(f)
        return model, class_names
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        return None, None

def organize_dataset():
    """
    Check dataset structure and organize if necessary.
    The Smart Wardrobe dataset typically has images already organized by category.
    """
    print("Checking dataset structure...")
    
    # Check if images are already organized in subdirectories
    if os.path.exists(TRAIN_DIR):
        subdirs = [d for d in os.listdir(TRAIN_DIR) 
                   if os.path.isdir(os.path.join(TRAIN_DIR, d))]
        
        if subdirs:
            print(f"✓ Found {len(subdirs)} clothing categories:")
            total_images = 0
            for subdir in sorted(subdirs):
                subdir_path = os.path.join(TRAIN_DIR, subdir)
                image_count = len([f for f in os.listdir(subdir_path) 
                                 if f.endswith(('.jpg', '.jpeg', '.png', '.JPG', '.JPEG', '.PNG'))])
                print(f"  - {subdir}: {image_count} images")
                total_images += image_count
            
            print(f"\n✓ Total: {total_images} images")
            print(f"✓ Dataset already organized!")
            
            # Return class names
            class_names = sorted(subdirs)
            return class_names, total_images > 0
    
    # If images are not organized, check if they're in a single folder
    if os.path.exists(TRAIN_DIR):
        image_files = [f for f in os.listdir(TRAIN_DIR) 
                      if f.endswith(('.jpg', '.jpeg', '.png', '.JPG', '.JPEG', '.PNG'))]
        
        if image_files:
            print(f"Found {len(image_files)} images in a flat structure.")
            print("Note: The Smart Wardrobe dataset typically comes pre-organized by category.")
            print("Please ensure your dataset has images organized in category folders.")
            return [], False
    
    print("⚠️  No training data found in the specified directory.")
    print("Please download the dataset from Kaggle and extract it.")
    return [], False

def get_class_names():
    """
    Get class names from the organized directory structure
    """
    if os.path.exists(TRAIN_DIR):
        subdirs = [d for d in os.listdir(TRAIN_DIR) 
                  if os.path.isdir(os.path.join(TRAIN_DIR, d))]
        return sorted(subdirs)
    return []

def create_cnn_model(num_classes):
    """
    Create a Convolutional Neural Network for multi-class classification
    """
    model = models.Sequential([
        # First Convolutional Block
        layers.Conv2D(32, (3, 3), activation='relu', input_shape=(IMG_SIZE, IMG_SIZE, 3)),
        layers.MaxPooling2D((2, 2)),
        
        # Second Convolutional Block
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        
        # Third Convolutional Block
        layers.Conv2D(128, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        
        # Fourth Convolutional Block
        layers.Conv2D(128, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        
        # Flatten and Dense Layers
        layers.Flatten(),
        layers.Dropout(0.5),  # Prevent overfitting
        layers.Dense(512, activation='relu'),
        layers.Dense(num_classes, activation='softmax')  # Multi-class classification
    ])
    
    return model

def prepare_data(class_names):
    """
    Prepare training and validation data with augmentation
    """
    # Data augmentation for training
    train_datagen = ImageDataGenerator(
        rescale=1./255,
        rotation_range=40,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        validation_split=0.2  # 20% for validation
    )
    
    # Only rescaling for validation
    val_datagen = ImageDataGenerator(
        rescale=1./255,
        validation_split=0.2
    )
    
    # Training generator
    train_generator = train_datagen.flow_from_directory(
        TRAIN_DIR,
        target_size=(IMG_SIZE, IMG_SIZE),
        batch_size=BATCH_SIZE,
        class_mode='categorical',  # Multi-class mode
        subset='training',
        classes={cls: idx for idx, cls in enumerate(class_names)}
    )
    
    # Validation generator
    validation_generator = val_datagen.flow_from_directory(
        TRAIN_DIR,
        target_size=(IMG_SIZE, IMG_SIZE),
        batch_size=BATCH_SIZE,
        class_mode='categorical',  # Multi-class mode
        subset='validation',
        classes={cls: idx for idx, cls in enumerate(class_names)}
    )
    
    return train_generator, validation_generator

def plot_training_history(history):
    """
    Plot training and validation accuracy/loss
    """
    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    
    epochs_range = range(len(acc))
    
    plt.figure(figsize=(12, 4))
    
    # Plot accuracy
    plt.subplot(1, 2, 1)
    plt.plot(epochs_range, acc, label='Training Accuracy')
    plt.plot(epochs_range, val_acc, label='Validation Accuracy')
    plt.legend(loc='lower right')
    plt.title('Training and Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    
    # Plot loss
    plt.subplot(1, 2, 2)
    plt.plot(epochs_range, loss, label='Training Loss')
    plt.plot(epochs_range, val_loss, label='Validation Loss')
    plt.legend(loc='upper right')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    
    plt.tight_layout()
    plt.savefig('training_history.png')
    plt.show()

def predict_image(model, image_path, class_names):
    """
    Predict clothing category for an image
    """
    try:
        # Validate image
        if not os.path.exists(image_path):
            print(f"❌ Error: File '{image_path}' not found")
            return None, None
        
        # Load and preprocess image
        img = keras.preprocessing.image.load_img(
            image_path, 
            target_size=(IMG_SIZE, IMG_SIZE)
        )
        img_array = keras.preprocessing.image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)
        img_array /= 255.0
        
        # Make prediction
        predictions = model.predict(img_array, verbose=0)[0]
        predicted_class_idx = np.argmax(predictions)
        predicted_class = class_names[predicted_class_idx]
        confidence = predictions[predicted_class_idx]
        
        # Get top 3 predictions
        top_3_indices = np.argsort(predictions)[-3:][::-1]
        
        # Display image
        plt.figure(figsize=(8, 6))
        plt.imshow(keras.preprocessing.image.load_img(image_path))
        plt.axis('off')
        
        emoji_map = {
            'shirt': '👔', 'top': '👕', 'dress': '👗', 'pants': '👖',
            'skirt': '👗', 'jacket': '🧥', 'coat': '🧥', 'shoe': '👞',
            'shoes': '👞', 'sweater': '🧶', 'hoodie': '🧥', 'hat': '🎩',
            'bag': '👜', 'accessory': '✨'
        }
        
        emoji = emoji_map.get(predicted_class.lower(), '👕')
        
        plt.title(
            f"{emoji} {predicted_class.upper()}\nConfidence: {confidence:.2%}",
            fontsize=16,
            fontweight='bold',
            color='darkblue'
        )
        
        plt.tight_layout()
        plt.savefig('prediction_result.png')
        plt.show()
        
        # Print detailed predictions
        print(f"\n{'='*50}")
        print(f"🔍 Prediction Results for: {os.path.basename(image_path)}")
        print(f"{'='*50}")
        print(f"Primary Prediction: {emoji} {predicted_class.upper()}")
        print(f"Confidence: {confidence:.2%}\n")
        print("Top 3 Predictions:")
        for i, idx in enumerate(top_3_indices, 1):
            print(f"  {i}. {class_names[idx]}: {predictions[idx]:.2%}")
        print(f"{'='*50}\n")
        
        return predicted_class, confidence
        
    except Exception as e:
        print(f"❌ Error predicting image: {e}")
        return None, None

def batch_predict(model, image_paths, class_names):
    """
    Predict multiple images at once
    """
    print(f"\n📊 Batch Prediction Mode - Processing {len(image_paths)} images\n")
    
    results = []
    for idx, image_path in enumerate(image_paths, 1):
        print(f"[{idx}/{len(image_paths)}] Analyzing: {image_path}")
        result, confidence = predict_image(model, image_path, class_names)
        
        if result:
            results.append({
                'image': image_path,
                'prediction': result,
                'confidence': confidence
            })
        
        print()
    
    # Summary
    if results:
        print(f"\n{'='*60}")
        print("📋 BATCH PREDICTION SUMMARY")
        print(f"{'='*60}")
        for idx, res in enumerate(results, 1):
            print(f"{idx}. {os.path.basename(res['image'])}")
            print(f"   ➜ Prediction: {res['prediction']} ({res['confidence']:.2%})")
        print(f"{'='*60}\n")
    
    return results

def interactive_prediction(model, class_names):
    """
    Interactive mode for testing images
    """
    print("\n" + "=" * 60)
    print("🔮 Interactive Prediction Mode - Clothing Classifier")
    print("=" * 60)
    print("\nYou can now test your model with clothing images!")
    print("Available categories:")
    for idx, cls in enumerate(class_names, 1):
        print(f"  {idx}. {cls}")
    print("\nOptions:")
    print("  - Enter single image path for one prediction")
    print("  - Enter 'batch' for multiple images")
    print("  - Enter 'quit' to exit\n")
    
    while True:
        user_input = input("📁 Enter image path (or command): ").strip()
        
        if user_input.lower() in ['quit', 'exit', 'q']:
            print("Exiting prediction mode...")
            break
        
        if user_input.lower() == 'batch':
            print("\n📂 Batch Mode - Enter image paths (one per line, 'done' to finish)")
            image_paths = []
            while True:
                path = input("  Image path: ").strip()
                if path.lower() == 'done':
                    break
                image_paths.append(path)
            
            if image_paths:
                batch_predict(model, image_paths, class_names)
            continue
        
        if not os.path.exists(user_input):
            print(f"❌ Error: File '{user_input}' not found. Try again.\n")
            continue
        
        print("\n🔍 Analyzing image...")
        result, confidence = predict_image(model, user_input, class_names)
        
        if result:
            print(f"✅ Predicted Category: {result}")
            print(f"📊 Confidence: {confidence:.2%}")
            print(f"💾 Result saved as: prediction_result.png\n")
        
        continue_pred = input("Test another image? (y/n): ").strip().lower()
        if continue_pred != 'y':
            break
    
    print("\n👋 Thanks for using the Clothing Classifier!")

def train_model(class_names):
    """
    Train a new model
    """
    num_classes = len(class_names)
    print(f"\n1. Creating CNN model for {num_classes}-class classification...")
    model = create_cnn_model(num_classes)
    model.summary()
    
    # Compile model
    print("\n2. Compiling model...")
    model.compile(
        optimizer='adam',
        loss='categorical_crossentropy',  # Multi-class loss
        metrics=['accuracy']
    )
    
    # Prepare data
    print("\n3. Preparing data...")
    try:
        train_generator, validation_generator = prepare_data(class_names)
        
        print(f"Found {train_generator.samples} training images")
        print(f"Found {validation_generator.samples} validation images")
        
        # Train model
        print(f"\n4. Training model for {EPOCHS} epochs...")
        history = model.fit(
            train_generator,
            epochs=EPOCHS,
            validation_data=validation_generator,
            verbose=1
        )
        
        # Save model
        print("\n5. Saving model...")
        model.save(MODEL_PATH)
        print(f"✓ Model saved as '{MODEL_PATH}'")
        
        # Save class names
        with open(CLASS_NAMES_PATH, 'w') as f:
            json.dump(class_names, f)
        print(f"✓ Class names saved as '{CLASS_NAMES_PATH}'")
        
        # Plot results
        print("\n6. Plotting training history...")
        plot_training_history(history)
        
        print("\n" + "=" * 60)
        print("✅ Training Complete!")
        print("=" * 60)
        print(f"\n✓ Model saved: {MODEL_PATH}")
        print(f"✓ Class names saved: {CLASS_NAMES_PATH}")
        print("✓ Training history plot saved: training_history.png")
        
        # Final metrics
        final_train_acc = history.history['accuracy'][-1]
        final_val_acc = history.history['val_accuracy'][-1]
        print(f"\n📊 Final Results:")
        print(f"   Training Accuracy: {final_train_acc:.2%}")
        print(f"   Validation Accuracy: {final_val_acc:.2%}")
        print(f"\n📁 Clothing Categories ({num_classes}):")
        for idx, cls in enumerate(class_names, 1):
            print(f"   {idx}. {cls}")
        
        return model
        
    except Exception as e:
        print(f"\n❌ Error during training: {e}")
        print("\nPlease ensure:")
        print("1. Download the dataset from Kaggle")
        print("2. Dataset should have images organized in category subfolders")
        print("3. Update TRAIN_DIR path in the code if needed")
        print("\nExpected structure:")
        print("  train/")
        print("    ├── shirts/")
        print("    ├── dresses/")
        print("    ├── pants/")
        print("    └── ... (other categories)")
        return None

def main():
    """
    Main execution function
    """
    print("=" * 60)
    print("🧠 Smart Wardrobe Clothing Classifier")
    print("=" * 60)
    
    # SMART MODEL CHECK - First thing we do!
    print("\n🔍 Checking for existing trained model...\n")
    
    if check_model_exists():
        print(f"✅ Found trained model: {MODEL_PATH}")
        model, class_names = load_saved_model()
        
        if model is not None and class_names is not None:
            print(f"✓ Model loaded successfully!")
            print(f"✓ Loaded {len(class_names)} clothing categories")
            print("\n" + "=" * 60)
            print("Model is ready! Skipping training and going to predictions...")
            print("=" * 60)
            
            # Start interactive prediction immediately
            interactive_prediction(model, class_names)
            return
    
    # No model found - need to train
    print("⚠️  No trained model found.")
    print("\nChecking for training data...")
    
    # Organize dataset and get class names
    print("\n0. Checking dataset structure...")
    class_names, success = organize_dataset()
    
    if not success or not class_names:
        print("\n❌ Error: Unable to find organized training data")
        print("\nPlease ensure:")
        print("1. Download the dataset from Kaggle")
        print("2. Extract it with clothing images organized in category subfolders")
        print("3. Update TRAIN_DIR path in the code if needed")
        print("\nExpected structure:")
        print("  train/")
        print("    ├── category1/")
        print("    │   ├── image1.jpg")
        print("    │   └── image2.jpg")
        print("    ├── category2/")
        print("    │   └── image3.jpg")
        return
    
    num_classes = len(class_names)
    print(f"\n✓ Found {num_classes} clothing categories")
    
    # Train new model
    model = train_model(class_names)
    
    if model is not None:
        # Ask if user wants to test the model
        print("\n" + "=" * 60)
        test_now = input("Would you like to test the model now? (y/n): ").strip().lower()
        if test_now == 'y':
            interactive_prediction(model, class_names)

if __name__ == "__main__":
    main()
