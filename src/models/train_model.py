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
    Trains all models and registers each one to the MLflow Model Registry.
    """
    print("--- Starting training and model registration ---")
    
    # Autolog will handle logging all parameters, metrics, and the model artifact
    mlflow.autolog()

    data_params = params['data']
    train_params = params['train']
    
    df = pd.read_csv(data_params['processed_path'])
    X = df.drop(columns=[data_params['target_column']])
    y = df[data_params['target_column']]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=data_params['test_size'], random_state=data_params['random_state']
    )

    for model_name in train_params['model_params']:
        # By adding 'as run', we can get the run's information, like its ID
        with mlflow.start_run(run_name=model_name) as run:
            print(f"\n--- Training and Logging: {model_name} ---")

            model_class = MODEL_MAPPING[model_name]
            hyperparameters = train_params['model_params'][model_name]
            
            model = model_class(**hyperparameters, random_state=data_params['random_state'])
            pipeline = Pipeline(steps=[("classifier", model)])
            
            pipeline.fit(X_train, y_train)

            # --- KEY CHANGE: REGISTER THE MODEL ---
            # 1. Get the URI of the model that autolog just saved
            model_uri = f"runs:/{run.info.run_id}/model"
            
            # 2. Create a unique name for the model in the registry
            # Example: "titanic_random_forest"
            registered_model_name = f"titanic_{model_name.replace(' ', '_').lower()}"
            
            print(f"Registering model to registry as: '{registered_model_name}'")
            
            # 3. Register the model
            mlflow.register_model(model_uri=model_uri, name=registered_model_name)

            print(f"--- Run for {model_name} complete and model registered. ---")
        model_save_path = Path(train_params['save_path'])

    model_save_path.parent.mkdir(exist_ok=True, parents=True)
    joblib.dump(pipeline, model_save_path) # <-- This was the line that saved the model

    print(f"Pipeline saved to: {model_save_path}")
    print("\n--- All training runs are complete. ---")

if __name__ == "__main__":
    mlflow.set_experiment("Titanic Survival Prediction")
    params = dvc.api.params_show()
    run_training(params=params)