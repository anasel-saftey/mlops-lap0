from pathlib import Path

import hydra
import joblib
import pandas as pd
from omegaconf import DictConfig, OmegaConf
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

# Import the updated preprocessor function
from src.features.build_features import create_preprocessor

# Map model names from the config to the actual sklearn classes
MODEL_MAPPING = {
    "Logistic Regression": LogisticRegression,
    "Random Forest": RandomForestClassifier,
}

@hydra.main(config_path="../../conf", config_name="config", version_base=None)
def run_training(cfg: DictConfig):
    """
    Runs a single training pipeline based on the loaded Hydra configuration.
    """
    print("--- Starting training with the following configuration: ---")
    print(OmegaConf.to_yaml(cfg))

    original_cwd = Path(hydra.utils.get_original_cwd())
    data_path = original_cwd / cfg.data.raw_path

    # --- Data Loading ---
    df = pd.read_csv(data_path)
    X = df.drop(columns=[cfg.data.target_column] + list(cfg.data.drop_columns))
    y = df[cfg.data.target_column]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=cfg.data.test_size, random_state=cfg.data.random_state
    )

    # --- Pipeline Creation ---
    preprocessor = create_preprocessor(cfg)

    # --- Model Training (for a single model) ---
    model_name = cfg.model.name
    model_class = MODEL_MAPPING[model_name]
    model = model_class(**cfg.model.params, random_state=cfg.data.random_state)
    full_pipeline = Pipeline(steps=[("preprocessor", preprocessor), ("classifier", model)])

    print(f"\n--- Training {model_name} ---")git
    full_pipeline.fit(X_train, y_train)

    # --- Evaluation ---
    y_pred = full_pipeline.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Accuracy for {model_name}: {accuracy:.4f}")

    # --- Model Saving ---
    model_save_path = original_cwd / cfg.evaluate.save_path
    model_save_path.parent.mkdir(exist_ok=True, parents=True)
    joblib.dump(full_pipeline, model_save_path)
    print(f"Pipeline for {model_name} saved to: {model_save_path}")
    print("\n--- Training pipeline finished. ---")


if __name__ == "__main__":
    run_training()