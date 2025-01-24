# Emotion Detection

This project uses a unique **Multi-branch CNN-BiLSTM model** to detect human emotions from facial images. The architecture is designed for real-time and robust emotion detection, combining spatial and temporal features.

## Features
- **Hybrid Architecture**: Combines CNN and BiLSTM for spatial and temporal feature extraction.
- **Transfer Learning**: Uses ResNet50 for pre-trained weights in the CNN branch.
- **Data Augmentation**: Enhances robustness with advanced augmentation techniques.
- **Real-Time Detection**: Integrates OpenCV for live emotion recognition.

## Installation
1. Clone the repository:
   ```bash
   git clone https://github.com/YourUsername/Emotion_Detection.git
   cd Emotion_Detection
## Training
Train the model:
```bash
python src/train.py --data_path data/raw/fer2013.csv
```
## Live Emotion Detection
Run real-time emotion detection:
```bash
python src/predict.py --model_path models/emotion_model.h5
```


