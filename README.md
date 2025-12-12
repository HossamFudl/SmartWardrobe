# 🧠 Smart Wardrobe Clothing Classifier

A deep learning-powered clothing classification system using Convolutional Neural Networks (CNN) to automatically identify and categorize clothing items from images.

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange)
![License](https://img.shields.io/badge/License-MIT-green)
![Status](https://img.shields.io/badge/Status-Active-brightgreen)

## ✨ Features

- 🤖 **Smart Model Detection** - Automatically checks for trained models and loads them instantly (saves training time!)
- 📸 **Single Image Prediction** - Classify individual clothing items with confidence scores
- 📂 **Batch Processing** - Analyze multiple images at once with detailed summary reports
- 🎯 **Multi-Class Classification** - Supports unlimited clothing categories
- 📊 **Top 3 Predictions** - Shows alternative predictions with probabilities
- 🎨 **Rich Visualization** - Displays images with prediction results and emojis
- 💾 **Model Persistence** - Saves trained models for reuse
- 📈 **Training History Plots** - Visualizes accuracy and loss over epochs
- ⚡ **Optimized Performance** - Fast predictions with data augmentation

## 🚀 Quick Start

### Prerequisites

```bash
Python 3.8 or higher
pip or conda package manager
```

### Installation

1. **Clone the repository:**
```bash
git clone https://github.com/yourusername/smart-wardrobe-classifier.git
cd smart-wardrobe-classifier
```

2. **Create a virtual environment (recommended):**
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. **Install dependencies:**
```bash
pip install -r requirements.txt
```

### Dataset Setup

1. **Download the dataset** from [Kaggle Smart Wardrobe Dataset](https://www.kaggle.com/datasets/hizkiasiregar/smart-wardrobe-clothing-dataset)

2. **Extract and organize:**
```
project-root/
├── clothing_classifier.py
├── train/
│   ├── shirts/
│   │   ├── image1.jpg
│   │   ├── image2.jpg
│   │   └── ...
│   ├── dresses/
│   │   ├── image1.jpg
│   │   └── ...
│   ├── pants/
│   └── ... (other categories)
```

### Usage

**First Run (Training):**
```bash
python clothing_classifier.py
```
- Detects dataset structure
- Creates and trains CNN model
- Saves model and class names
- Launches interactive prediction mode

**Subsequent Runs (Fast Predictions):**
```bash
python clothing_classifier.py
```
- ✅ Finds existing model
- ⚡ Loads instantly (no training needed!)
- 🎯 Goes straight to predictions

## 📖 Usage Modes

### Single Image Prediction

```
📁 Enter image path (or command): /path/to/shirt.jpg

🔍 Prediction Results for: shirt.jpg
==================================================
Primary Prediction: 👔 SHIRT
Confidence: 95.23%

Top 3 Predictions:
  1. shirt: 95.23%
  2. top: 3.45%
  3. jacket: 1.32%
==================================================
```

### Batch Mode (Multiple Images)

```
📁 Enter image path (or command): batch

📂 Batch Mode - Enter image paths (one per line, 'done' to finish)
  Image path: photo1.jpg
  Image path: photo2.jpg
  Image path: photo3.jpg
  Image path: done

📋 BATCH PREDICTION SUMMARY
================================================== ==
1. photo1.jpg
   ➜ Prediction: shirt (95.23%)
2. photo2.jpg
   ➜ Prediction: dress (87.45%)
3. photo3.jpg
   ➜ Prediction: pants (92.10%)
================================================== ==
```

### Interactive Commands

| Command | Description |
|---------|-------------|
| `/path/to/image.jpg` | Predict single image |
| `batch` | Enter batch prediction mode |
| `quit` / `exit` / `q` | Exit the program |

## 🏗️ Architecture

### CNN Model Structure

```
Input: 150x150x3 RGB Image
    ↓
Conv2D (32 filters, 3x3) → ReLU → MaxPool
    ↓
Conv2D (64 filters, 3x3) → ReLU → MaxPool
    ↓
Conv2D (128 filters, 3x3) → ReLU → MaxPool
    ↓
Conv2D (128 filters, 3x3) → ReLU → MaxPool
    ↓
Flatten → Dropout (0.5)
    ↓
Dense (512, ReLU) → Dense (N_classes, Softmax)
    ↓
Output: Class Probabilities
```

### Key Components

- **Convolutional Blocks**: Extract visual features from images
- **Max Pooling**: Reduce spatial dimensions while preserving important information
- **Dropout**: Prevent overfitting during training
- **Softmax**: Multi-class probability distribution
- **Data Augmentation**: Rotation, zoom, shift, flip for robust training

## ⚙️ Configuration

Edit these settings in `clothing_classifier.py`:

```python
IMG_SIZE = 150           # Image dimension (150x150 pixels)
BATCH_SIZE = 32          # Training batch size
EPOCHS = 25              # Number of training epochs
TRAIN_DIR = 'train'      # Path to training data
```

## 📊 Output Files

After training, the following files are generated:

| File | Description |
|------|-------------|
| `clothing_classifier_model.h5` | Trained CNN model |
| `class_names.json` | List of clothing categories |
| `training_history.png` | Accuracy and loss plots |
| `prediction_result.png` | Last prediction visualization |

## 🎯 Model Performance

The model achieves good accuracy across multiple clothing categories:

- **Training Accuracy**: ~85-95% (varies by dataset)
- **Validation Accuracy**: ~80-90%
- **Inference Time**: ~100-200ms per image
- **Model Size**: ~25-30 MB

*Performance depends on dataset size and quality*

## 🔧 Advanced Usage

### Custom Dataset

To use your own dataset:

1. Organize images in category folders under `train/`
2. Update `TRAIN_DIR` path if needed
3. Run the script - it auto-detects categories

### Adjust Training Parameters

```python
EPOCHS = 50              # More epochs for better accuracy
BATCH_SIZE = 16          # Smaller batch size for limited memory
IMG_SIZE = 224           # Larger images for more detail
```

### Transfer Learning (Optional)

For faster training with pre-trained models:

```python
base_model = keras.applications.MobileNetV2(input_shape=(IMG_SIZE, IMG_SIZE, 3))
base_model.trainable = False
# Add custom layers on top
```

## 📋 Project Structure

```
smart-wardrobe-classifier/
├── clothing_classifier.py      # Main application
├── requirements.txt            # Python dependencies
├── README.md                   # Documentation
├── train/                      # Training data folder
│   ├── shirts/
│   ├── dresses/
│   ├── pants/
│   └── ...
└── (Generated files after running)
    ├── clothing_classifier_model.h5
    ├── class_names.json
    ├── training_history.png
    └── prediction_result.png
```

## 🛠️ Dependencies

```
tensorflow==2.13.0
keras==2.13.0
numpy>=1.21.0
matplotlib>=3.4.0
pillow>=8.0.0
```

See `requirements.txt` for complete list.

## 🔍 Troubleshooting

### Issue: "No training data found"

**Solution**: Ensure your dataset is organized correctly:
```
train/
├── category1/
│   ├── image1.jpg
│   └── image2.jpg
└── category2/
    └── image1.jpg
```

### Issue: "Model not loading"

**Solution**: Delete `clothing_classifier_model.h5` and `class_names.json`, then retrain.

### Issue: "CUDA out of memory"

**Solution**: Reduce `BATCH_SIZE` to 16 or 8 in the configuration.

### Issue: "Low accuracy"

**Solution**:
- Increase `EPOCHS` (e.g., 50 or 100)
- Ensure dataset quality and sufficient images per category (100+ per category)
- Add more data augmentation

## 📈 Performance Tips

1. **Dataset Quality**: Use clear, well-lit images of clothing items
2. **Category Balance**: Keep similar number of images per category
3. **Training Time**: More epochs = better accuracy (diminishing returns after ~50)
4. **GPU**: Use GPU for faster training (automatically detected by TensorFlow)

## 🤝 Contributing

Contributions are welcome! Here's how:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit changes (`git commit -m 'Add amazing feature'`)
4. Push to branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📝 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 👤 Author

**Your Name** - [@yourtwitter](https://twitter.com/yourtwitter) - your.email@example.com

## 🙏 Acknowledgments

- **TensorFlow & Keras** - Deep learning framework
- **Kaggle** - Smart Wardrobe Dataset provider
- **Open Source Community** - For amazing tools and libraries

## 📚 Resources

- [TensorFlow Documentation](https://www.tensorflow.org/guide)
- [CNN Fundamentals](https://cs231n.github.io/)
- [Kaggle Datasets](https://www.kaggle.com/datasets)
- [Deep Learning Best Practices](https://www.deeplearningbook.org/)

## ⭐ Show Your Support

Give a ⭐ if this project helped you! It motivates development.

## 📞 Support

For questions or issues:
- Open an [GitHub Issue](https://github.com/yourusername/smart-wardrobe-classifier/issues)
- Email: your.email@example.com
- Check existing [Discussions](https://github.com/yourusername/smart-wardrobe-classifier/discussions)

---

**Last Updated**: December 2025  
**Version**: 1.0.0  
**Status**: ✅ Active & Maintained
