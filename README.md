# Emotion Detection with CNN

![emotion_detection](https://github.com/datamagic2020/Emotion_detection_with_CNN/blob/main/emoition_detection.png)

This project trains and runs a CNN-based facial emotion detector using TensorFlow/Keras and OpenCV.

## What's modernized

- Replaced deprecated `keras` imports with `tensorflow.keras`
- Replaced deprecated `fit_generator()` and `predict_generator()` calls
- Removed the hardcoded Windows video path from the test script
- Added CLI arguments for dataset paths, model paths, and runtime options
- Added support for saving a modern `.keras` model while still exporting legacy JSON and weights files

## Recommended environment

- Python 3.11 is recommended
- Windows users should prefer a fresh virtual environment instead of the checked-in `venv/` folders

TensorFlow does not currently support every Python release. If you are on Python 3.13, create a Python 3.11 environment before installing dependencies.

## Install

```bash
python -m venv .venv
.venv\Scripts\activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

## Dataset

Download the FER2013 dataset from [Kaggle](https://www.kaggle.com/msambare/fer2013) and place it in this structure:

```text
data/
  train/
  test/
```

## Train

```bash
python TrainEmotionDetector.py --data-dir data --model-dir model --epochs 50
```

The training script saves:

- `model/emotion_model.keras`
- `model/emotion_model.json`
- `model/emotion_model.weights.h5`

## Run detection

Use your webcam:

```bash
python TestEmotionDetector.py --source 0
```

Use a local video file:

```bash
python TestEmotionDetector.py --source meme.mp4
```

Press `q` to quit the preview window.

## Evaluate

```bash
python EvaluateEmotionDetector.py --data-dir data --model-dir model
```

## Notes

- Haar cascade file path defaults to `haarcascades/haarcascade_frontalface_default.xml`
- The runtime scripts will load `model/emotion_model.keras` first, then fall back to legacy files if needed
