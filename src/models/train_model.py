import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
import dvc.api
import mlflow
import mlflow.sklearn
from mlflow.tracking import MlflowClient
import yaml
import os
import joblib # Make sure joblib is imported
from pathlib import Path # Make sure Path is imported

MODEL_MAPPING = {
    "Logistic Regression": LogisticRegression,
    "Random Forest": RandomForestClassifier,
}

def run_training(params: dict):
    """
    Manually logs training, registers models, promotes the best one,
    and saves the best model pipeline to a local file.
    """
    print("--- Starting training with full manual logging ---")

    data_params = params['data']
    train_params = params['train']
    
    df = pd.read_csv(data_params['processed_path'])
    X = df.drop(columns=[data_params['target_column']])
    y = df[data_params['target_column']]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=data_params['test_size'], random_state=data_params['random_state']
    )

    run_results = {}

    for model_name in train_params['model_params']:
        with mlflow.start_run(run_name=model_name) as run:
            run_id = run.info.run_id
            print(f"\n--- Training: {model_name} (Run ID: {run_id}) ---")

            model_class = MODEL_MAPPING[model_name]
            hyperparameters = train_params['model_params'][model_name]
            
            mlflow.log_params(hyperparameters)
            mlflow.log_param("model_name", model_name)

            pipeline = Pipeline(steps=[("classifier", model_class(**hyperparameters, random_state=data_params['random_state']))])
            pipeline.fit(X_train, y_train)

            y_pred = pipeline.predict(X_test)
            accuracy = accuracy_score(y_test, y_pred)
            mlflow.log_metric("accuracy", float(accuracy))

            # To fix the warning in your screenshot, add an input_example
            # input_example = X_train.head(1)
            # mlflow.sklearn.log_model(
            #     sk_model=pipeline,
            #     name="model",
            #     input_example=input_example
            # )

            model_uri = 'https://dagshub.com/anasel-saftey/mlops-lap0.mlflow'
            registered_model_name = f"titanic_{model_name.replace(' ', '_').lower()}"
            mlflow.register_model(model_uri=model_uri, name=registered_model_name)
            
            # --- KEY CHANGE 1: Store the trained pipeline object ---
            run_results[run_id] = {
                'accuracy': accuracy,
                'model_name': registered_model_name,
                'pipeline_object': pipeline # Store the actual trained model
            }
            print(f"--- Run for {model_name} complete. Accuracy: {accuracy:.4f} ---")

    if run_results:
        # --- Find and Promote the Best Model in MLflow ---
        best_run_id = max(run_results, key=lambda k: run_results[k]['accuracy'])
        best_accuracy = run_results[best_run_id]['accuracy']
        best_model_name = run_results[best_run_id]['model_name']
        
        print(f"\n--- Best model is '{best_model_name}' with accuracy: {best_accuracy:.4f} ---")
        # --- KEY CHANGE 2: Save the Best Model Pipeline Locally ---
        print(f"\n--- Saving best model pipeline locally ---")
        best_pipeline = run_results[best_run_id]['pipeline_object']
        model_save_path = Path(train_params['save_path'])
        
        # Ensure the directory exists
        model_save_path.parent.mkdir(exist_ok=True, parents=True)
        
        joblib.dump(best_pipeline, model_save_path)
        print(f"✅ Best model pipeline saved to: {model_save_path}")

if __name__ == "__main__":
    mlflow.set_tracking_uri('https://dagshub.com/anasel-saftey/mlops-lap0.mlflow')
    os.environ['MLFLOW_TRACKING_USERNAME'] = 'anasel-saftey'
    os.environ['MLFLOW_TRACKING_PASSWORD'] = '739b995aa6fc36b3ed1d653603a44e7819d57797'
    mlflow.set_experiment("Titanic Survival Prediction")
    params = dvc.api.params_show()
    run_training(params=params)