from __future__ import annotations

import argparse
from pathlib import Path

import cv2
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator

from emotion_model_utils import DEFAULT_MODEL_DIR, KERAS_MODEL_NAME, build_emotion_model, save_legacy_model_files


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the CNN-based emotion detector.")
    parser.add_argument("--data-dir", type=Path, default=Path("data"), help="Dataset root containing train/ and test/.")
    parser.add_argument("--model-dir", type=Path, default=DEFAULT_MODEL_DIR, help="Directory used to save trained models.")
    parser.add_argument("--epochs", type=int, default=50, help="Number of training epochs.")
    parser.add_argument("--batch-size", type=int, default=64, help="Mini-batch size.")
    parser.add_argument("--learning-rate", type=float, default=1e-4, help="Adam learning rate.")
    return parser.parse_args()


def create_generators(data_dir: Path, batch_size: int):
    train_dir = data_dir / "train"
    validation_dir = data_dir / "test"

    if not train_dir.exists() or not validation_dir.exists():
        raise FileNotFoundError(
            f"Expected dataset folders at {train_dir.resolve()} and {validation_dir.resolve()}."
        )

    train_data_gen = ImageDataGenerator(rescale=1.0 / 255)
    validation_data_gen = ImageDataGenerator(rescale=1.0 / 255)

    train_generator = train_data_gen.flow_from_directory(
        train_dir,
        target_size=(48, 48),
        batch_size=batch_size,
        color_mode="grayscale",
        class_mode="categorical",
    )

    validation_generator = validation_data_gen.flow_from_directory(
        validation_dir,
        target_size=(48, 48),
        batch_size=batch_size,
        color_mode="grayscale",
        class_mode="categorical",
    )
    return train_generator, validation_generator


def main() -> None:
    args = parse_args()
    cv2.ocl.setUseOpenCL(False)

    train_generator, validation_generator = create_generators(args.data_dir, args.batch_size)
    emotion_model = build_emotion_model()
    emotion_model.compile(
        loss="categorical_crossentropy",
        optimizer=Adam(learning_rate=args.learning_rate),
        metrics=["accuracy"],
    )

    emotion_model.fit(
        train_generator,
        epochs=args.epochs,
        validation_data=validation_generator,
    )

    args.model_dir.mkdir(parents=True, exist_ok=True)
    keras_model_path = args.model_dir / KERAS_MODEL_NAME
    emotion_model.save(keras_model_path)
    save_legacy_model_files(emotion_model, args.model_dir)

    print(f"Saved modern model to: {keras_model_path.resolve()}")
    print(f"Saved legacy compatibility files to: {args.model_dir.resolve()}")


if __name__ == "__main__":
    main()
