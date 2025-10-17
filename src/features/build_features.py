import pandas as pd
from pathlib import Path
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
import dvc.api # Import the DVC API

def create_preprocessor(params: dict) -> ColumnTransformer:
    """Creates a ColumnTransformer from parameters."""
    numeric_features = params['features']['numeric_features']
    categorical_features = params['features']['categorical_features']

    numeric_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="mean")),
        ("scaler", StandardScaler()),
    ])
    categorical_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ])

    return ColumnTransformer(transformers=[
        ("num", numeric_transformer, numeric_features),
        ("cat", categorical_transformer, categorical_features),
    ], remainder="drop")

def process_features(params: dict): # Accept params as an argument
    """Loads raw data, processes features, and saves the result."""
    print("--- Starting feature processing ---")
    
    # The params dictionary is now passed in directly from the main block
    input_path = params['data']['raw_path']
    output_path = params['data']['processed_path']
    target_column = params['data']['target_column']
    
    df_raw = pd.read_csv(input_path)
    if target_column in df_raw.columns:
        X_raw = df_raw.drop(columns=[target_column])
        y = df_raw[[target_column]]
    else:
        X_raw = df_raw
        y = None

    preprocessor = create_preprocessor(params)
    X_processed_np = preprocessor.fit_transform(X_raw)
    
    ohe_feature_names = preprocessor.named_transformers_['cat']['onehot'].get_feature_names_out(params['features']['categorical_features'])
    processed_feature_names = list(params['features']['numeric_features']) + list(ohe_feature_names)
    X_processed = pd.DataFrame(X_processed_np, columns=processed_feature_names, index=df_raw.index)
    
    if y is not None:
        df_processed = pd.concat([X_processed, y], axis=1)
    else:
        df_processed = X_processed
    
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    df_processed.to_csv(output_path, index=False)
    
    print(f"Processed data saved to: {output_path}")
    print("--- Feature processing finished ---")

if __name__ == "__main__":
    # Use the DVC API to load the parameters DVC is tracking
    params = dvc.api.params_show()
    process_features(params=params)
