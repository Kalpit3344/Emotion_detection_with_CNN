from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np

from emotion_model_utils import DEFAULT_MODEL_DIR, EMOTION_LABELS, load_emotion_model


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run emotion detection on a webcam or video file.")
    parser.add_argument(
        "--source",
        default="0",
        help="Video source. Use 0 for webcam, or pass a local video file path such as meme.mp4.",
    )
    parser.add_argument("--model-dir", type=Path, default=DEFAULT_MODEL_DIR, help="Directory that stores the trained model.")
    parser.add_argument(
        "--cascade-path",
        type=Path,
        default=Path("haarcascades") / "haarcascade_frontalface_default.xml",
        help="Path to the Haar cascade XML file.",
    )
    parser.add_argument("--width", type=int, default=1280, help="Preview width.")
    parser.add_argument("--height", type=int, default=720, help="Preview height.")
    return parser.parse_args()


def open_capture(source: str) -> cv2.VideoCapture:
    if source.isdigit():
        return cv2.VideoCapture(int(source))
    return cv2.VideoCapture(source)


def main() -> None:
    args = parse_args()
    emotion_model = load_emotion_model(args.model_dir)
    print(f"Loaded model from {args.model_dir.resolve()}")

    face_detector = cv2.CascadeClassifier(str(args.cascade_path))
    if face_detector.empty():
        raise FileNotFoundError(f"Could not load Haar cascade from {args.cascade_path.resolve()}")

    cap = open_capture(args.source)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video source: {args.source}")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.resize(frame, (args.width, args.height))
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        num_faces = face_detector.detectMultiScale(gray_frame, scaleFactor=1.3, minNeighbors=5)

        for (x, y, w, h) in num_faces:
            cv2.rectangle(frame, (x, y - 50), (x + w, y + h + 10), (0, 255, 0), 4)
            roi_gray_frame = gray_frame[y : y + h, x : x + w]
            cropped_img = cv2.resize(roi_gray_frame, (48, 48)).astype("float32") / 255.0
            cropped_img = np.expand_dims(cropped_img, axis=(0, -1))

            emotion_prediction = emotion_model.predict(cropped_img, verbose=0)
            maxindex = int(np.argmax(emotion_prediction))
            cv2.putText(
                frame,
                EMOTION_LABELS[maxindex],
                (x + 5, y - 20),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (255, 0, 0),
                2,
                cv2.LINE_AA,
            )

        cv2.imshow("Emotion Detection", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
