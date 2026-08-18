from __future__ import annotations

from pathlib import Path

from tensorflow.keras import Sequential
from tensorflow.keras.layers import Conv2D, Dense, Dropout, Flatten, Input, MaxPooling2D
from tensorflow.keras.models import load_model, model_from_json


EMOTION_LABELS = {
    0: "Angry",
    1: "Disgusted",
    2: "Fearful",
    3: "Happy",
    4: "Neutral",
    5: "Sad",
    6: "Surprised",
}

DEFAULT_MODEL_DIR = Path("model")
KERAS_MODEL_NAME = "emotion_model.keras"
LEGACY_JSON_NAME = "emotion_model.json"
LEGACY_WEIGHTS_NAME = "emotion_model.weights.h5"
LEGACY_COMPAT_WEIGHTS_NAME = "emotion_model.h5"


def build_emotion_model() -> Sequential:
    model = Sequential(
        [
            Input(shape=(48, 48, 1)),
            Conv2D(32, kernel_size=(3, 3), activation="relu"),
            Conv2D(64, kernel_size=(3, 3), activation="relu"),
            MaxPooling2D(pool_size=(2, 2)),
            Dropout(0.25),
            Conv2D(128, kernel_size=(3, 3), activation="relu"),
            MaxPooling2D(pool_size=(2, 2)),
            Conv2D(128, kernel_size=(3, 3), activation="relu"),
            MaxPooling2D(pool_size=(2, 2)),
            Dropout(0.25),
            Flatten(),
            Dense(1024, activation="relu"),
            Dropout(0.5),
            Dense(len(EMOTION_LABELS), activation="softmax"),
        ]
    )
    return model


def load_emotion_model(model_dir: Path | str = DEFAULT_MODEL_DIR):
    model_dir = Path(model_dir)
    keras_model_path = model_dir / KERAS_MODEL_NAME
    legacy_json_path = model_dir / LEGACY_JSON_NAME
    legacy_weights_paths = [
        model_dir / LEGACY_WEIGHTS_NAME,
        model_dir / LEGACY_COMPAT_WEIGHTS_NAME,
    ]

    if keras_model_path.exists():
        return load_model(keras_model_path)

    if legacy_json_path.exists():
        for weights_path in legacy_weights_paths:
            if weights_path.exists():
                try:
                    model = model_from_json(legacy_json_path.read_text(encoding="utf-8"))
                    model.load_weights(weights_path)
                    return model
                except (TypeError, ValueError):
                    # Keras 3 can reject older JSON configs; rebuild the known architecture instead.
                    model = build_emotion_model()
                    model.load_weights(weights_path)
                    return model
        raise FileNotFoundError(
            f"Found {legacy_json_path}, but no matching weights file was present in {model_dir}."
        )

    raise FileNotFoundError(
        "No trained model was found. Expected one of these files inside "
        f"{model_dir.resolve()}: {KERAS_MODEL_NAME}, {LEGACY_JSON_NAME}, "
        f"{LEGACY_WEIGHTS_NAME}, or {LEGACY_COMPAT_WEIGHTS_NAME}."
    )


def save_legacy_model_files(model, model_dir: Path | str = DEFAULT_MODEL_DIR) -> None:
    model_dir = Path(model_dir)
    model_dir.mkdir(parents=True, exist_ok=True)

    json_path = model_dir / LEGACY_JSON_NAME
    weights_path = model_dir / LEGACY_WEIGHTS_NAME

    json_path.write_text(model.to_json(), encoding="utf-8")
    model.save_weights(weights_path)
