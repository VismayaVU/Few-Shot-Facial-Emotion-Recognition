# 🤖 Real-Time Facial Expression Detection using Few-Shot Learning

This project implements a **Prototypical Network-based facial expression recognition system** that uses few-shot learning principles to classify human facial expressions in real time. It is built using the FER2013 dataset and leverages the `easyfsl` framework along with PyTorch and OpenCV.

## 👨‍💻 Team Members

- Shri Hari S - PES1UG22AM154  
- Venkat Subramanian - PES1UG22AM188  
- Vishwanath Sridhar - PES1UG22AM194  
- Vismaya Vadana - PES1UG22AM195  

## 🧠 Project Summary

- Uses **Prototypical Networks** to perform few-shot learning for facial emotion recognition.
- Trained and evaluated on the **FER2013** dataset.
- Implements **live webcam-based prediction** of facial expressions.
- Applies **data augmentation** for robust training.
- Smooths predictions in real-time using a moving average strategy.

## 📦 Libraries & Tools Used

- Python
- PyTorch
- torchvision
- easyfsl
- OpenCV
- FER2013 dataset
- PIL, NumPy

## 📁 Dataset

- FER2013 facial expression dataset is downloaded from Kaggle.
- The dataset includes images categorized by emotions like Happy, Angry, Sad, Neutral, etc.

## 🧪 Few-Shot Learning Setup

- **N-Way**: 5
- **K-Shot (Support)**: 10 images per class
- **Query**: 15 images per class
- **Training Episodes**: 60,000
- **Evaluation Tasks**: 100

## 🔧 Data Preprocessing

Includes:

- Resize to 48x48
- Random horizontal flips, rotation, jitter
- Normalization using ImageNet statistics

## 🧩 Model Architecture

### Prototypical Networks

- Backbone: `ResNet18` (pretrained on ImageNet, FC replaced with flattening)
- Projection Head: `512 -> 256 -> 128`
- Classification based on Euclidean distance between query embeddings and class prototypes

## 🎯 Evaluation

- Tested over 100 evaluation tasks
- Accuracy is computed using query set predictions
- Early stopping is used to prevent overfitting
- Model checkpoint (`best_model.pth`) saved during training

## 🎥 Real-Time Inference

- Captures video stream using OpenCV
- Detects faces using Haar cascades
- Uses pre-computed support set to perform classification using embeddings and prototypes
- Prediction smoothing is done using majority vote over last 10 frames

## 🛠️ Installation

```bash
pip install torch torchvision easyfsl opencv-python pillow kagglehub
```
- Ensure you have a webcam connected for live testing.

## ▶️ Running the Project

- Download the FER2013 dataset via KaggleHub.
- Run the training loop or load the pretrained model (best_model.pth).
- Start the webcam to begin live emotion detection.

## 💾 Model Files

- best_model.pth: Saved model checkpoint from training
- model.pth: Used for final evaluation and real-time inference

## 🙏 Acknowledgments

- FER2013 Dataset on Kaggle
- easyfsl
- PyTorch and torchvision communities
