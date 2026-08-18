from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay, classification_report, confusion_matrix
from tensorflow.keras.preprocessing.image import ImageDataGenerator

from emotion_model_utils import DEFAULT_MODEL_DIR, EMOTION_LABELS, load_emotion_model


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate the trained emotion detector on the test split.")
    parser.add_argument("--data-dir", type=Path, default=Path("data"), help="Dataset root containing test/.")
    parser.add_argument("--model-dir", type=Path, default=DEFAULT_MODEL_DIR, help="Directory that stores the trained model.")
    parser.add_argument("--batch-size", type=int, default=64, help="Mini-batch size used for evaluation.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    test_dir = args.data_dir / "test"
    if not test_dir.exists():
        raise FileNotFoundError(f"Expected test dataset at {test_dir.resolve()}")

    emotion_model = load_emotion_model(args.model_dir)
    print(f"Loaded model from {args.model_dir.resolve()}")

    test_data_gen = ImageDataGenerator(rescale=1.0 / 255)
    test_generator = test_data_gen.flow_from_directory(
        test_dir,
        target_size=(48, 48),
        batch_size=args.batch_size,
        color_mode="grayscale",
        class_mode="categorical",
        shuffle=False,
    )

    predictions = emotion_model.predict(test_generator, verbose=1)
    predicted_classes = predictions.argmax(axis=1)
    label_names = [EMOTION_LABELS[index] for index in sorted(EMOTION_LABELS)]

    print("-----------------------------------------------------------------")
    c_matrix = confusion_matrix(test_generator.classes, predicted_classes)
    print(c_matrix)
    cm_display = ConfusionMatrixDisplay(confusion_matrix=c_matrix, display_labels=label_names)
    cm_display.plot(cmap=plt.cm.Blues)
    plt.tight_layout()
    plt.show()

    print("-----------------------------------------------------------------")
    print(
        classification_report(
            test_generator.classes,
            predicted_classes,
            target_names=label_names,
            digits=4,
        )
    )


if __name__ == "__main__":
    main()
