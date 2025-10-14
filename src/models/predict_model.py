# src/models/predict_model.py

from pathlib import Path

import joblib
import pandas as pd


def make_prediction():
    """
    Loads the trained pipeline and makes a prediction on sample data.
    """
    print("Making a prediction...")

    PROJECT_ROOT = Path(__file__).resolve().parents[2]
    MODEL_PATH = PROJECT_ROOT / "models" / "titanic_best_pipeline.joblib"

    # --- Load Model ---
    if not MODEL_PATH.exists():
        print(f"Error: Model file not found at {MODEL_PATH}")
        print(
            "Please run the training script first: `python -m src.models.train_model`"
        )
        return

    # Load the entire pipeline (preprocessor + model)
    loaded_pipeline = joblib.load(MODEL_PATH)
    print("Model pipeline loaded successfully.")

    # --- Prepare Sample Data for Prediction ---
    # This data simulates a new passenger or a batch of new passengers.
    # The structure must match the training data's columns.
    sample_data = pd.DataFrame(
        {
            "Pclass": [3, 1],
            "Sex": ["male", "female"],
            "Age": [22.0, 38.0],
            "SibSp": [1, 1],
            "Parch": [0, 0],
            "Fare": [7.25, 71.2833],
            "Embarked": ["S", "C"],
        }
    )

    print("\nSample Data:")
    print(sample_data)

    # --- Make Predictions ---
    predictions = loaded_pipeline.predict(sample_data)
    prediction_proba = loaded_pipeline.predict_proba(sample_data)

    # --- Display Results ---
    results_df = sample_data.copy()
    results_df["Predicted_Survived_Flag"] = predictions
    # Use .apply() for a more robust mapping from 0/1 to No/Yes
    results_df["Prediction"] = results_df["Predicted_Survived_Flag"].apply(
        lambda x: "Yes" if x == 1 else "No"
    )
    results_df["Survival_Probability_%"] = prediction_proba[:, 1] * 100

    print("\nPrediction Results:")
    print(
        results_df[
            ["Sex", "Age", "Pclass", "Prediction", "Survival_Probability_%"]
        ].round(2)
    )


if __name__ == "__main__":
    # To run this script directly, you must run it as a module from the project root
    # Example: python -m src.models.predict_model
    make_prediction()
