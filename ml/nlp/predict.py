from pathlib import Path
import joblib

PROJECT_ROOT = Path(__file__).resolve().parents[2]
MODEL_PATH = PROJECT_ROOT / "models" / "text_classifier.joblib"

model = joblib.load(MODEL_PATH)


def predict_message(test: str, context: str | None = None) -> str:
    model_input = f"Conversation context:\n{context}\n\nCurrent message:\n{test}" if context else test
    prediction = model.predict([model_input])[0]

    return prediction


if __name__ == "__main__":
    message = "how can you sleap with make up on your face, you will wake up while you are oily "
    result = predict_message(message)
    print(f"message: {message}")
    print(f"Predict category: {result}")
