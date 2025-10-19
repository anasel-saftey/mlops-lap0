import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
import dvc.api
import mlflow
import mlflow.sklearn
import yaml
import joblib
from pathlib import Path

MODEL_MAPPING = {
    "Logistic Regression": LogisticRegression,
    "Random Forest": RandomForestClassifier,
}

def run_training(params: dict):
    """
    Loads data, trains ALL models specified in params, and uses MLflow's autologging.
    """
    print("--- Starting training with autologging enabled ---")
    
    # Enable MLflow Autologging
    mlflow.autolog()

    data_params = params['data']
    train_params = params['train']
    
    df = pd.read_csv(data_params['processed_path'])
    X = df.drop(columns=[data_params['target_column']])
    y = df[data_params['target_column']]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=data_params['test_size'], random_state=data_params['random_state']
    )

    # This loop iterates over ALL models listed in the `model_params` section of your YAML
    for model_name in train_params['model_params']:
        with mlflow.start_run(run_name=model_name):
            print(f"\n--- Training: {model_name} ---")

            model_class = MODEL_MAPPING[model_name]
            hyperparameters = train_params['model_params'][model_name]
            
            model = model_class(**hyperparameters, random_state=data_params['random_state'])
            pipeline = Pipeline(steps=[("classifier", model)])
            
            print(f"Fitting {model_name}...")
            pipeline.fit(X_train, y_train)

            # Print accuracy for confirmation in the console
            y_pred = pipeline.predict(X_test)
            accuracy = accuracy_score(y_test, y_pred)
            print(f"Accuracy for {model_name}: {accuracy:.4f}")
            print(f"--- Run for {model_name} automatically logged. ---")


# ... inside the run_training function ...
    model_save_path = Path(train_params['save_path'])
    model_save_path.parent.mkdir(exist_ok=True, parents=True)
    joblib.dump(pipeline, model_save_path) # <-- This was the line that saved the model

    print(f"Pipeline saved to: {model_save_path}")
    print("\n--- All training runs are complete. ---")

if __name__ == "__main__":
    mlflow.set_experiment("Titanic Survival Prediction")
    params = dvc.api.params_show()  
    run_training(params=params)