import warnings  # Used to suppress non-essential warnings for cleaner output.
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder

from sklearn.dummy import DummyRegressor  # Baseline regression model for comparison.
from sklearn.tree import DecisionTreeRegressor  # Decision tree regression model.

from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score  # Regression evaluation metrics.


# -----------------------------
# DECISION TREE REGRESSION MODEL
# -----------------------------

MILLION = 10**6  # Used to display money-related errors in millions.
RANDOM_STATE = 42  # Fixed seed ensures reproducible splits and results.
warnings.filterwarnings("ignore")  # Suppress warnings to keep printed output readable.


def print_metrics(tag, y_true, y_pred):
    """Print R2, RMSE, and MAE in a consistent format for model comparison."""
    r2 = r2_score(y_true, y_pred)  # R^2 measures explained variance
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))  # RMSE measures typical error magnitude
    mae = mean_absolute_error(y_true, y_pred)  # MAE measures average absolute error
    print(f"{tag}: R2={r2:.4f} RMSE={rmse / MILLION:.2f}M MAE={mae / MILLION:.2f}M")


def make_preprocess(numeric_cols, categorical_cols):
    """Create preprocessing that imputes missing values and one-hot encodes categorical features."""
    return ColumnTransformer(
        transformers=[
            ("numeric", SimpleImputer(strategy="mean"), numeric_cols),  # Replace missing numeric values with mean.
            ("categorical", Pipeline([
                ("imputer", SimpleImputer(strategy="constant", fill_value="Unknown")),  # Fill missing categories.
                ("onehot", OneHotEncoder(handle_unknown="ignore", drop="first")),  # One-hot encode with safe handling.
            ]), categorical_cols),
        ],
        remainder="drop"  # Drop any columns not explicitly processed.
    )


def main():
    # 1) Load dataset
    file_path = "data/Mojo_budget_update.csv"  # Dataset location in the repository structure.
    data = pd.read_csv(file_path)  # Read CSV into a DataFrame.


    # 2) Feature extraction: run_time_minutes
    runtime_text = data["run_time"].fillna("")  # Replace missing run_time text with empty strings.
    hours = runtime_text.str.extract(r"(\d+)\s*hr", expand=False).fillna(0).astype(int)
    minutes = runtime_text.str.extract(r"(\d+)\s*min", expand=False).fillna(0).astype(int)
    data["run_time_minutes"] = hours * 60 + minutes  # Convert runtime to total minutes.


    # 3) Drop columns not intended for modelling
    data = data.drop(columns=[
        "movie_id", "title", "trivia", "html",
        "release_date", "run_time",
        "distributor", "director", "writer", "producer",
        "composer", "cinematographer",
        "main_actor_1", "main_actor_2", "main_actor_3", "main_actor_4"
    ])


    # 4) Define target and features
    data["worldwide"] = pd.to_numeric(data["worldwide"], errors="coerce")  # Ensure target is numeric.
    data = data.dropna(subset=["worldwide"])  # Remove rows where the target is missing.

    y = data["worldwide"]  # Target variable
    X = data.drop(columns=["worldwide", "domestic", "international"])  # Features used for prediction.


    # 5) Identify numeric vs categorical columns (needed for ColumnTransformer)
    categorical_cols = ["genre_1", "genre_2", "genre_3", "genre_4", "mpaa"]  # Known categorical columns.
    numeric_cols = X.select_dtypes(include=["number"]).columns.tolist()  # All numeric feature columns.


    # 6) Split data into train/validation/test
    #    25% test
    #    75% train+val -> 60% train and 15% val overall
    X_trainval, X_test, y_trainval, y_test = train_test_split(
        X, y, test_size=0.25, random_state=RANDOM_STATE
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_trainval, y_trainval, test_size=0.20, random_state=RANDOM_STATE
    )


    # 7) Baseline model (DummyRegressor) using the same preprocessing pipeline
    baseline = Pipeline([
        ("preprocess", make_preprocess(numeric_cols, categorical_cols)),  # Apply imputation + encoding.
        ("model", DummyRegressor(strategy="mean")),  # Predict the mean of y
    ])
    baseline.fit(X_train, y_train)  # Train baseline on training set only.

    y_base_val = baseline.predict(X_val)  # Predict on validation set.
    print_metrics("BASELINE_VAL", y_val, y_base_val)  # Print baseline metrics for validation.
    print("-" * 40)  # Visual separator for readability.


    # 8) Decision Tree model
    tree_model = Pipeline([
        ("preprocess", make_preprocess(numeric_cols, categorical_cols)),  # Apply same preprocessing for fairness.
        ("model", DecisionTreeRegressor(random_state=RANDOM_STATE)),  # Default tree; prevents randomness variability.
    ])
    tree_model.fit(X_train, y_train)  # Train decision tree on training data.

    # Evaluate on training and validation to observe overfitting tendency
    y_pred_train = tree_model.predict(X_train)
    y_pred_val = tree_model.predict(X_val)

    print_metrics("DT_TRAIN", y_train, y_pred_train)  # Train performance.
    print_metrics("DT_VAL", y_val, y_pred_val)  # Validation performance.


if __name__ == "__main__":
    main()  # Execute the training/evaluation workflow.
