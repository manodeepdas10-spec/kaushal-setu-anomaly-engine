import pandas as pd
import joblib
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

def build_production_pipeline(numeric_features: list, categorical_features: list) -> Pipeline:
    """Builds a scikit-learn pipeline capable of handling missing data and text features."""

    #1. Pipeline for numeric metrics (Salary, Retention, Job Charges)
    num_transformer = Pipeline([('imputer', SimpleImputer(strategy='median')), ('scaler', StandardScaler())])

    #2. Pipeline for categorical variables (District, Course Category)
    cat_transformer = Pipeline([('imputer', SimpleImputer(strategy='most_frequent')), ('encoder', OneHotEncoder(handle_unknown='ignore', sparse_output=False))])

    #3. Combined Preprocessor
    preprocessor = ColumnTransformer(transformers=[('num', num_transformer, numeric_features), ('cat', cat_transformer, categorical_features)])

    #4. Final Isolation Forest Anomaly Pipeline
    anomaly_pipeline = Pipeline([('preprocessor', preprocessor), ('model', IsolationForest(contamination=0.05, random_state=42, n_estimators=200))])

    return anomaly_pipeline

if __name__ == "__main__":
    # Load the real dataset CSV
    df_train = pd.read_csv("kaushal_setu_trainee_data.csv") # type: ignore

    num_cols = ['salary', 'retention_months', 'job_changes']
    cat_cols = ['district', 'course_category', 'qualification']

    # Train pipeline on CSV data
    pipeline = build_production_pipeline(num_cols, cat_cols)
    pipeline.fit(df_train)

    # Export model for FastAPI
    joblib.dump(pipeline, 'kaushal_setu_anomaly_model.pkl')
    print("Model successfully trained on CSV data and saved!")