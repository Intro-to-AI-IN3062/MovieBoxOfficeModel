import pandas as pd
from sklearn.dummy import DummyRegressor
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
import warnings

#SIMPLE LINEAR REGRESSION MODEL
MILLION = 10**6
RANDOM_STATE = 42
warnings.filterwarnings("ignore")

#Helper methods-------------------
def print_metrics(tag, y_true, y_pred):
        r2 = r2_score(y_true, y_pred)
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        mae = mean_absolute_error(y_true, y_pred)
        print(f"{tag}: R2={r2:.4f} RMSE={rmse/(MILLION):.2f}M MAE={mae/MILLION:.2f}M")

def make_preprocess():
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
#-------------------------------

file_path = 'data/Mojo_budget_update.csv'
data = pd.read_csv(file_path)

#Generating files for report
write_data_quality_csv(data, "Reports/LinearRegression/LR_Data_Quality.csv")
#---------------------------
#New runtime feature column
runtime_text = data['run_time'].fillna('')
hours = runtime_text.str.extract(r'(\d+)\s*hr', expand=False).fillna(0).astype(int)
minutes = runtime_text.str.extract(r'(\d+)\s*min', expand=False).fillna(0).astype(int)
data['run_time_minutes'] = hours * 60 + minutes

#DATA CLEANUP
data = data.drop(columns=[
            'movie_id', 'title', 'trivia', 'html',
            'release_date', 'run_time',
            'distributor', 'director', 'writer', 'producer',
            'composer', 'cinematographer',
            'main_actor_1', 'main_actor_2', 'main_actor_3', 'main_actor_4'
        ])
data["worldwide"] = pd.to_numeric(data["worldwide"], errors="coerce")
data = data.dropna(subset=["worldwide"])

y = data["worldwide"] #Target
X = data.drop(columns=['worldwide', 'domestic', 'international']) #Features
print("CHECK NA : ", X.isna().sum().sort_values(ascending=False).head(10))

#PREPROCESSING
categorical_cols = ['genre_1','genre_2','genre_3','genre_4','mpaa']
numeric_cols = X.select_dtypes(include=['number']).columns.tolist()

#SPLITTING DATA (train/validation/test) (same as Random Forest Regression)
'''
25% test
75% train+validation ->  60% train + 15% validation

'''
X_trainval, X_test, y_trainval, y_test = train_test_split(
    X, y, test_size=0.25, random_state=RANDOM_STATE
    )
X_train, X_val, y_train, y_val = train_test_split(
    X_trainval, y_trainval, test_size=0.20, random_state=RANDOM_STATE
)

lr_runs = []#stored runs for generating report

#BASELINE MODEL (uses pipeline) --------
baseline = Pipeline([
    ("preprocess", make_preprocess()),
    ("model", DummyRegressor(strategy="mean")),
])
baseline.fit(X_train, y_train)

y_base = baseline.predict(X_val)

#for report generation
y_base_train = baseline.predict(X_train)
base_train_m = calc_metrics(y_train, y_base_train)
base_val_m = calc_metrics(y_val, y_base)
record_run(lr_runs, "Baseline(DummyMean)", {"strategy": "mean"}, base_train_m, base_val_m)
#-----

print("BASELINE R^2:", r2_score(y_val, y_base))
print("BASELINE RMSE:", np.sqrt(mean_squared_error(y_val, y_base)))
print("BASELINE MAE:", mean_absolute_error(y_val, y_base))
print("-" * 30)
#---------------------------------------

#Training
param_sets = [ #for automated testing
    {"fit_intercept":True},
    {"fit_intercept":False},
]

#Evaluate each model
for p in param_sets:
    #Creating/Training the model (uses pipeline)
    model = Pipeline([
    ("preprocess", make_preprocess()),
    ("model", LinearRegression(
         fit_intercept=p["fit_intercept"],
         n_jobs=1
    ),)
])
    model.fit(X_train, y_train)
    print("USING PARAMS:", p)

    #Evaluating the model
    y_pred_train = model.predict(X_train) #Training
    y_pred_val = model.predict(X_val) #Validation

    #for report generation
    train_metrics = calc_metrics(y_train, y_pred_train)
    val_metrics = calc_metrics(y_val, y_pred_val) 
    record_run(lr_runs, "LinearRegression", p, train_metrics, val_metrics)
    #--------

    print_metrics("LR_TRAIN", y_train, y_pred_train)
    print_metrics("LR_VAL", y_val, y_pred_val)

    print("-" * 30)

#Final Evaluation of best model
best_p = {"fit_intercept": True}
best_model = Pipeline([
    ("preprocess", make_preprocess()),
    ("model", LinearRegression(
        fit_intercept=best_p["fit_intercept"],
        n_jobs=1,
    )),
])

write_runs_csv(lr_runs, "Reports/LinearRegression/LR_Tuning_Runs.csv") #GENERATE RUNS REPORT

best_model.fit(X_trainval, y_trainval) #refit model
y_pred_test = best_model.predict(X_test) #predict on x-test set
print_metrics("LR_TEST", y_test, y_pred_test)