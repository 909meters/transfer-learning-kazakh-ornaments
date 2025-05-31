# Kazakh Ornaments Classification

This project focuses on classifying Kazakh ornaments into three categories: **Flora**, **Geo**, and **Zoo** using a convolutional neural network (ResNet18).

## Project Overview

The goal is to automatically recognize and classify images of traditional Kazakh ornaments. This can be useful for cultural heritage preservation, educational tools, and digital archiving.

## Getting Started

1. **Install dependencies:**
   ```
   pip install -r requirements.txt
   ```

2. **Prepare the dataset:**
   - Place your images in the `Dataset` folder, organized by class.
   - Split the dataset:
     ```
     python split.py
     ```

3. **Train and evaluate the model:**
   ```
   python main.py
   ```

## Project Structure

- `main.py` — Main script for training, testing, and Grad-CAM visualization
- `data_loader.py` — Data loading and preprocessing
- `model_resnet.py` — Model architecture (ResNet18)
- `train_model.py` — Model training logic
- `evaluate_model.py` — Model evaluation and metrics
- `visualize_gradcam.py` — Grad-CAM visualization for model interpretability
- `split.py` — Script for splitting the dataset into train/val/test

## Technologies Used

- PyTorch
- torchvision
- split-folders
- OpenCV
- matplotlib, seaborn

## Author

Aida Kuatkyzy 
kuatkyzyaida@gmail.com 
908816@stud.unive.it