import os
import pandas as pd
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
#CENTRALISED HELPER METHODS FOR REPORT GENERATION / PRINTING METRICS


MILLION = 10**6

#Helper methods-------------------
def print_metrics(tag, y_true, y_pred):
    r2 = r2_score(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    print(f"{tag}: R2={r2:.4f} RMSE={rmse/(MILLION):.2f}M MAE={mae/MILLION:.2f}M")

def make_preprocess(numeric_cols, categorical_cols):
    return ColumnTransformer(
        transformers=[
            ("numeric", SimpleImputer(strategy="mean"), numeric_cols),
            ("categorical", Pipeline([
                ("imputer", SimpleImputer(strategy="constant", fill_value="Unknown")),
                ("onehot", OneHotEncoder(handle_unknown="ignore", drop="first")),
            ]), categorical_cols),
        ],
        remainder="drop"
    )


#REPORT GENERATING METHODS
#DATA QUALITY---------
def write_data_quality_csv(df, out_path): 
    missing_count = df.isna().sum()
    missing_count = missing_count[missing_count > 0]
    missing_count = missing_count.sort_values(ascending=False)
    #Build table
    out = pd.DataFrame()
    out["feature"] = missing_count.index
    out["missing_count"] = missing_count.values
    out["missing_%"] = ((out["missing_count"] / len(df)) * 100).round(2)

    out.to_csv(out_path, index=False)
    return out

#RUN RESULTS---------------
def calc_metrics(y_true, y_pred):
    return {
        "r2": float(r2_score(y_true, y_pred)),
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "mae": float(mean_absolute_error(y_true, y_pred)),
    }

def record_run(rows, model_name, params_dict, train_metrics, val_metrics):
    row = {
        "model": model_name,
        "params": str(params_dict),
        "r2_train": train_metrics["r2"],
        "rmse_train": train_metrics["rmse"],
        "mae_train": train_metrics["mae"],
        "r2_val": val_metrics["r2"],
        "rmse_val": val_metrics["rmse"],
        "mae_val": val_metrics["mae"],
        "r2_gap_train_minus_val": train_metrics["r2"] - val_metrics["r2"],
    }
    rows.append(row)

def write_runs_csv(rows, out_path="tuning_results.csv"):
    pd.DataFrame(rows).to_csv(out_path, index=False)