import joblib
import pandas as pd
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
import dvc.api # Import the DVC API

MODEL_MAPPING = {
    "Logistic Regression": LogisticRegression,
    "Random Forest": RandomForestClassifier,
}

def run_training(params: dict): # Accept params as an argument
    """Loads processed data and trains a model based on DVC parameters."""
    print("--- Starting training ---")

    # The params dictionary is now passed in directly from the main block
    data_params = params['data']
    train_params = params['train']
    
    df = pd.read_csv(data_params['processed_path'])
    X = df.drop(columns=[data_params['target_column']])
    y = df[data_params['target_column']]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=data_params['test_size'], random_state=data_params['random_state']
    )

    model_name = train_params['model_name']
    model_class = MODEL_MAPPING[model_name]
    hyperparameters = train_params['model_params'][model_name]
    
    model = model_class(**hyperparameters, random_state=data_params['random_state'])
    pipeline = Pipeline(steps=[("classifier", model)])
    
    print(f"\n--- Training {model_name} ---")
    pipeline.fit(X_train, y_train)

    y_pred = pipeline.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Accuracy for {model_name}: {accuracy:.4f}")

    model_save_path = Path(train_params['save_path'])
    model_save_path.parent.mkdir(exist_ok=True, parents=True)
    joblib.dump(pipeline, model_save_path)
    
    print(f"Pipeline saved to: {model_save_path}")
    print("\n--- Training finished ---")

if __name__ == "__main__":
    # Use the DVC API to load the parameters DVC is tracking
    params = dvc.api.params_show()
    run_training(params=params)

