# flower-identifier

This is an old project I made as part of Nittany AI's Machine Learning Bootcamp Course.

This is an old project I made as part of Nittany AI's Machine Learning Bootcamp Course.
What It Does
- Data Preparation & Augmentation:
    Uses TensorFlow’s ImageDataGenerator to load and augment a dataset of flower images organized by category.
- CNN Architecture:
    Implements a Convolutional Neural Network with multiple convolutional and pooling layers to learn visual features from flower images.
- Model Training & Evaluation:
    Trains the model using a split of training and validation data, outputs training metrics, and saves the trained model.
- Prediction Capabilities:
    Provides a sample script to load a test image and predict its class, demonstrating real-time inference.
- Modular Design:
    Easily configurable for different flower datasets, image dimensions, or class counts, making it a flexible starting point for further experimentation.

How to Get Started:
  Install the required libraries (e.g., TensorFlow, NumPy, Matplotlib, OpenCV, Pillow).
  Organize your dataset in a directory with subfolders for each flower class.
  Update file paths in the code as needed.
  Run the script using your preferred command line with:
  python flower_identifier.py
