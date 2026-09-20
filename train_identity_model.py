import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report

def build_identity_pipeline(numeric_features: list, categorical_features: list) -> Pipeline:
    """Create a production classification pipeline for fake trainee identification."""

    num_transformer = Pipeline([('imputer', SimpleImputer(strategy='mean')), ('scaler', StandardScaler())])

    cat_transformer = Pipeline([('imputer', SimpleImputer(strategy='constant', fill_value='UNKNOWN')), ('encoder', OneHotEncoder(handle_unknown='ignore', sparse_output=False))])

    preprocessor = ColumnTransformer(transformers=[('num', num_transformer, numeric_features), ('cat', cat_transformer, categorical_features)])

    # Balanced random forest ensures minority ghost profiles are accurately flagged
    identity_pipeline = Pipeline([('preprocessor', preprocessor), ('model', RandomForestClassifier(n_estimators=150, class_weight='balanced', random_state=42, max_depth=12))])

    return identity_pipeline

if __name__ == "__main__":
    # Load dataset
    df = pd.read_csv("kaushal_setu_identity_logs.csv")

    num_cols = ['module_completion_speed', 'device_concurrency_count', 'face_similarity_index']
    cat_cols = ['liveness_status', 'ip_risk_score', 'training_center_id']

    X = df.drop(columns=['is_fake_trainee'])
    y=df['is_fake_trainee']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train model
    pipeline = build_identity_pipeline(num_cols, cat_cols)
    pipeline.fit(X_train, y_train)

    # Evaluation verification
    preds = pipeline.predict(X_test)
    print("\n--- Identity Pipeline Classification Metrics ---")
    print(classification_report(y_test, preds, target_names=['Valid Trainee', 'Fake/Proxy Trainee']))

    # Export model for the FastAPI server
    joblib.dump(pipeline, 'kaushal_setu_identity_model.pkl')
    print("✨ Identity verification pipeline successfully exported!")