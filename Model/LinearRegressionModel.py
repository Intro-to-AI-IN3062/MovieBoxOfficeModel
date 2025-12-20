import pandas as pd
from sklearn.dummy import DummyRegressor
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import warnings
from model_reports import ( #Helper methods for report generation
    print_metrics, make_preprocess, write_data_quality_csv, calc_metrics, record_run, write_runs_csv
)
from data_prep import clean_data

#SIMPLE LINEAR REGRESSION MODEL
RANDOM_STATE = 42
warnings.filterwarnings("ignore")

file_path = 'data/Mojo_budget_update.csv'
data = pd.read_csv(file_path)

write_data_quality_csv(data, "Reports/LinearRegression/LR_Data_Quality.csv")#Includes only missing values for now
#---------------------------
#New runtime feature column
runtime_text = data['run_time'].fillna('')
hours = runtime_text.str.extract(r'(\d+)\s*hr', expand=False).fillna(0).astype(int)
minutes = runtime_text.str.extract(r'(\d+)\s*min', expand=False).fillna(0).astype(int)
data['run_time_minutes'] = hours * 60 + minutes

data, X, y, numeric_cols, categorical_cols = clean_data(data) #DATA CLEANUP (modularised now)

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
    ("preprocess", make_preprocess(numeric_cols, categorical_cols)),
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
    ("preprocess", make_preprocess(numeric_cols, categorical_cols)),
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
    ("preprocess", make_preprocess(numeric_cols, categorical_cols)),
    ("model", LinearRegression(
        fit_intercept=best_p["fit_intercept"],
        n_jobs=1,
    )),
])

write_runs_csv(lr_runs, "Reports/LinearRegression/LR_Tuning_Runs.csv") #GENERATE RUNS REPORT

best_model.fit(X_trainval, y_trainval) #refit model
y_pred_test = best_model.predict(X_test) #predict on x-test set
print_metrics("LR_TEST", y_test, y_pred_test)

#Write final results as table
test_results = calc_metrics(y_test, y_pred_test)
pd.DataFrame([{
    "model": "LinearRegression",
    "params": str(best_p),
    "r2_test": test_results["r2"],
    "rmse_test": test_results["rmse"],
    "mae_test": test_results["mae"],
}]).to_csv("Reports/LinearRegression/LR_Test_Result.csv", index=False) #FINAL RESULT FILE

#Write baseline table (only once from this file ONLY) ----
baseline.fit(X_trainval, y_trainval)
y_base_test = baseline.predict(X_test)
base_m = calc_metrics(y_test, y_base_test)

pd.DataFrame([{
    "model": "Baseline(DummyMean)",
    "params": "{'strategy':'mean'}",
    "r2_test": base_m["r2"],
    "rmse_test": base_m["rmse"],
    "mae_test": base_m["mae"],
}]).to_csv("Reports/Baseline_Test_Result.csv", index=False)
