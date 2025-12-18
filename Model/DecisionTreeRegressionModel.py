import warnings  # Used to suppress non-essential warnings for cleaner output
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder

from sklearn.dummy import DummyRegressor  # Baseline regression model for comparison
from sklearn.tree import DecisionTreeRegressor  # Decision tree regression model

from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score  # Regression evaluation metrics


# -----------------------------
# DECISION TREE REGRESSION MODEL
# -----------------------------

MILLION = 10**6  # Used to display money-related errors in millions
RANDOM_STATE = 42  # Fixed seed ensures reproducible splits and results
warnings.filterwarnings("ignore")  # Suppress warnings to keep printed output readable


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
            ("numeric", SimpleImputer(strategy="mean"), numeric_cols),  # Replace missing numeric values with mean
            ("categorical", Pipeline([
                ("imputer", SimpleImputer(strategy="constant", fill_value="Unknown")),  # Fill missing categories
                ("onehot", OneHotEncoder(handle_unknown="ignore", drop="first")),  # One-hot encode with safe handling
            ]), categorical_cols),
        ],
        remainder="drop"  # Drop any columns not explicitly processed
    )


def main():
    # 1) Load dataset
    file_path = "data/Mojo_budget_update.csv"  # Dataset location in the repository structure
    data = pd.read_csv(file_path)  # Read CSV into a DataFrame


    # 2) Feature extraction: run_time_minutes
    runtime_text = data["run_time"].fillna("")  # Replace missing run_time text with empty strings
    hours = runtime_text.str.extract(r"(\d+)\s*hr", expand=False).fillna(0).astype(int)
    minutes = runtime_text.str.extract(r"(\d+)\s*min", expand=False).fillna(0).astype(int)
    data["run_time_minutes"] = hours * 60 + minutes  # Convert runtime to total minutes


    # 3) Drop columns not intended for modelling
    data = data.drop(columns=[
        "movie_id", "title", "trivia", "html",
        "release_date", "run_time",
        "distributor", "director", "writer", "producer",
        "composer", "cinematographer",
        "main_actor_1", "main_actor_2", "main_actor_3", "main_actor_4"
    ])


    # 4) Define target and features
    data["worldwide"] = pd.to_numeric(data["worldwide"], errors="coerce")  # Ensure target is numeric
    data = data.dropna(subset=["worldwide"])  # Remove rows where the target is missing

    y = data["worldwide"]  # Target variable
    X = data.drop(columns=["worldwide", "domestic", "international"])  # Features used for prediction


    # 5) Identify numeric vs categorical columns (needed for ColumnTransformer)
    categorical_cols = ["genre_1", "genre_2", "genre_3", "genre_4", "mpaa"]  # Known categorical columns
    numeric_cols = X.select_dtypes(include=["number"]).columns.tolist()  # All numeric feature columns


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
        ("preprocess", make_preprocess(numeric_cols, categorical_cols)),  # Apply imputation + encoding
        ("model", DummyRegressor(strategy="mean")),  # Predict the mean of y
    ])
    baseline.fit(X_train, y_train)  # Train baseline on training set only

    y_base_val = baseline.predict(X_val)  # Predict on validation set
    print_metrics("BASELINE_VAL", y_val, y_base_val)  # Print baseline metrics for validation
    print("-" * 40)  # Visual separator for readability


    # 8) Decision Tree model: systematic hyperparameter tuning
    #    We tune on validation only and keep the test set untouched until final evaluation
    param_grid = {
        "max_depth": [None, 10, 20, 30],          # Controls tree complexity
        "min_samples_split": [2, 10],             # Minimum samples required to split an internal node
        "min_samples_leaf": [1, 5, 10],           # Minimum samples required to be at a leaf node
        "max_features": [None, "sqrt"],           # Feature subsampling can reduce variance
    }

    tuning_rows = []  # Each row stores one hyperparameter setting plus train/val metrics

    # Loop through all combinations
    for max_depth in param_grid["max_depth"]:
        for min_samples_split in param_grid["min_samples_split"]:
            for min_samples_leaf in param_grid["min_samples_leaf"]:
                for max_features in param_grid["max_features"]:

                    # Build a pipeline so preprocessing is identical for every model trial (fair comparison)
                    model = Pipeline([
                        ("preprocess", make_preprocess(numeric_cols, categorical_cols)),
                        ("model", DecisionTreeRegressor(
                            random_state=RANDOM_STATE,            # Fix randomness for reproducibility
                            max_depth=max_depth,                  # Limit depth to control complexity
                            min_samples_split=min_samples_split,  # Require more samples to split
                            min_samples_leaf=min_samples_leaf,    # Require more samples in each leaf
                            max_features=max_features             # Restrict features considered per split
                        )),
                    ])

                    # Train on training set only
                    model.fit(X_train, y_train)

                    # Predict on training and validation sets to detect overfitting
                    y_pred_train = model.predict(X_train)
                    y_pred_val = model.predict(X_val)

                    # Compute metrics for training
                    r2_train = r2_score(y_train, y_pred_train)
                    rmse_train = np.sqrt(mean_squared_error(y_train, y_pred_train))
                    mae_train = mean_absolute_error(y_train, y_pred_train)

                    # Compute metrics for validation
                    r2_val = r2_score(y_val, y_pred_val)
                    rmse_val = np.sqrt(mean_squared_error(y_val, y_pred_val))
                    mae_val = mean_absolute_error(y_val, y_pred_val)

                    # Store results in a structured format for reporting
                    tuning_rows.append({
                        "max_depth": max_depth,
                        "min_samples_split": min_samples_split,
                        "min_samples_leaf": min_samples_leaf,
                        "max_features": max_features,
                        "train_r2": r2_train,
                        "train_rmse": rmse_train,
                        "train_mae": mae_train,
                        "val_r2": r2_val,
                        "val_rmse": rmse_val,
                        "val_mae": mae_val,
                    })


    # Convert results to a DataFrame and rank by validation RMSE
    tuning_df = pd.DataFrame(tuning_rows)
    tuning_df = tuning_df.sort_values(by="val_rmse", ascending=True).reset_index(drop=True)

    # Save tuning results for the report / observations file evidence
    results_path = "DecisionTree_TuningResults.csv"
    tuning_df.to_csv(results_path, index=False)

    # Print a small summary of top 10
    print("Top 10 DecisionTree tuning results (sorted by VAL RMSE):")
    display_cols = [
        "max_depth", "min_samples_split", "min_samples_leaf", "max_features",
        "train_r2", "val_r2", "train_rmse", "val_rmse", "train_mae", "val_mae"
    ]
    print(tuning_df[display_cols].head(10).to_string(index=False))
    print("-" * 40)

    # Identify the best hyperparameter set (lowest validation RMSE)
    best_row = tuning_df.iloc[0].to_dict()
    print("Best params (by VAL RMSE):")

    # Pandas can convert None/int columns into float with NaN
    # DecisionTreeRegressor requires max_depth to be int or None
    best_max_depth = best_row["max_depth"]
    if pd.isna(best_max_depth):  # If NaN, treat it as None
        best_max_depth = None
    else:
        best_max_depth = int(best_max_depth)  # Convert float to int

    best_min_samples_split = int(best_row["min_samples_split"])
    best_min_samples_leaf = int(best_row["min_samples_leaf"])
    best_max_features = best_row["max_features"]

    print({
        "max_depth": best_max_depth,
        "min_samples_split": best_min_samples_split,
        "min_samples_leaf": best_min_samples_leaf,
        "max_features": best_max_features,
    })

    print("Best validation performance:")

    # Build the best model pipeline once, then evaluate it (avoids inline fit/predict type issues)
    best_model = Pipeline([
        ("preprocess", make_preprocess(numeric_cols, categorical_cols)),
        ("model", DecisionTreeRegressor(
            random_state=RANDOM_STATE,
            max_depth=best_max_depth,
            min_samples_split=best_min_samples_split,
            min_samples_leaf=best_min_samples_leaf,
            max_features=best_max_features,
        )),
    ])

    best_model.fit(X_train, y_train)
    y_best_val = best_model.predict(X_val)
    print_metrics("DT_VAL_BEST", y_val, y_best_val)


if __name__ == "__main__":
    main()
