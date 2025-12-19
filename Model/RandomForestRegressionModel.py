import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.dummy import DummyRegressor
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
import warnings
from model_reports import ( #Helper methods for report generation
    print_metrics, make_preprocess, write_data_quality_csv, calc_metrics, record_run, write_runs_csv
)
from data_prep import clean_data

#RANDOM FOREST REGRESSION MODEL
RANDOM_STATE = 42
warnings.filterwarnings("ignore")

#Import data
file_path = 'data/Mojo_budget_update.csv'
data = pd.read_csv(file_path)
write_data_quality_csv(data, "Reports/RandomForest/RF_Data_Quality.csv")

data, X, y, numeric_cols, categorical_cols = clean_data(data) #DATA CLEANUP (modularised now)

#SPLITTING DATA (train/validation/test)
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

rf_runs = []#stored runs for generating report

#BASELINE MODEL (uses pipeline)
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
record_run(rf_runs, "Baseline(DummyMean)", {"strategy": "mean"}, base_train_m, base_val_m)
#-----

print("BASELINE R^2:", r2_score(y_val, y_base))
print("BASELINE RMSE:", np.sqrt(mean_squared_error(y_val, y_base)))
print("BASELINE MAE:", mean_absolute_error(y_val, y_base))
print("-" * 30)
#--------------------

#Training/Evaluating
param_sets = [ #for automated testing
    {"n_estimators": 200, "max_depth": None, "min_samples_leaf": 1,  "max_features": "sqrt"},
    {"n_estimators": 400, "max_depth": 30,   "min_samples_leaf": 5,  "max_features": "sqrt"},
    {"n_estimators": 600, "max_depth": 20,   "min_samples_leaf": 10, "max_features": "sqrt"},
    {"n_estimators": 400, "max_depth": 15,   "min_samples_leaf": 20, "max_features": "sqrt"},
]

for p in param_sets:
    #Creating/Training the model (uses pipeline)
    model = Pipeline([
    ("preprocess", make_preprocess(numeric_cols, categorical_cols)),
    ("model", RandomForestRegressor(
        n_estimators=p['n_estimators'],
        max_depth=p['max_depth'],
        min_samples_leaf=p["min_samples_leaf"],
        max_features=p["max_features"],
        random_state=RANDOM_STATE,
        n_jobs=1,
    )),
])
    model.fit(X_train, y_train)
    print("USING PARAMS:", p)

    #Evaluating the model
    y_pred_train = model.predict(X_train) #Training
    y_pred_val   = model.predict(X_val) #Validation

    #for report generation
    train_metrics = calc_metrics(y_train, y_pred_train)
    val_metrics = calc_metrics(y_val, y_pred_val) 
    record_run(rf_runs, "RandomForest", p, train_metrics, val_metrics)
    #--------

    print_metrics("RF_TRAIN", y_train, y_pred_train)
    print_metrics("RF_VAL",   y_val,   y_pred_val)

    print("-" * 30)

#Final Evaluation of best model
best_p = {"n_estimators": 200, "max_depth": None, "min_samples_leaf": 1, "max_features": "sqrt"}
best_model = Pipeline([
    ("preprocess", make_preprocess(numeric_cols, categorical_cols)),
    ("model", RandomForestRegressor(
        n_estimators=best_p["n_estimators"],
        max_depth=best_p["max_depth"],
        min_samples_leaf=best_p["min_samples_leaf"],
        max_features=best_p["max_features"],
        random_state=RANDOM_STATE,
        n_jobs=1,
    )),
])

write_runs_csv(rf_runs, "Reports/RandomForest/RF_Tuning_Runs.csv") #GENERATE RUNS REPORT

best_model.fit(X_trainval, y_trainval) #refit model
y_pred_test = best_model.predict(X_test) #predict on x-test set
print_metrics("RF_TEST", y_test, y_pred_test)

#FINAL RESULTS
test_results = calc_metrics(y_test, y_pred_test)
pd.DataFrame([{
    "model": "RandomForest",
    "params": str(best_p),
    "r2_test": test_results["r2"],
    "rmse_test": test_results["rmse"],
    "mae_test": test_results["mae"],
}]).to_csv("Reports/RandomForest/RF_Test_Result.csv", index=False)