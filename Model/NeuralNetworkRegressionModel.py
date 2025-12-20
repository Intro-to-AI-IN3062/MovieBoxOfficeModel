from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras import regularizers
from tensorflow.keras.layers import Input
from tensorflow.keras.models import load_model
from tensorflow.keras.callbacks import EarlyStopping

import numpy as np
import pandas as pd
import io
import os
import requests
from sklearn import metrics
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler 
from sklearn.dummy import DummyRegressor
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
import warnings
from model_reports import ( #Helper methods for report generation
    print_metrics, make_preprocess, write_data_quality_csv, calc_metrics, record_run, write_runs_csv
)
from data_prep import clean_data

#NEURAL NETWORK REGRESSION MODEL
RANDOM_STATE = 42
warnings.filterwarnings("ignore")

file_path = 'data/Mojo_budget_update.csv'
data = pd.read_csv(file_path)
write_data_quality_csv(data, "Reports/NeuralNetwork/NN_Data_Quality.csv")

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

preprocessor = make_preprocess(numeric_cols, categorical_cols)
X_train = preprocessor.fit_transform(X_train)
X_val = preprocessor.transform(X_val)
X_test = preprocessor.transform(X_test)
X_trainval = preprocessor.fit_transform(X_trainval)

nn_runs = []  # Store runs for reporting

#BASELINE MODEL
baseline = DummyRegressor(strategy="mean")
baseline.fit(X_train, y_train)

y_base = baseline.predict(X_val)

#for report generation
y_base_train = baseline.predict(X_train)
base_train_m = calc_metrics(y_train, y_base_train)
base_val_m = calc_metrics(y_val, y_base)
record_run(nn_runs, "Baseline(DummyMean)", {"strategy": "mean"}, base_train_m, base_val_m)
#-----

print("BASELINE R^2:", r2_score(y_val, y_base))
print("BASELINE RMSE:", np.sqrt(mean_squared_error(y_val, y_base)))
print("BASELINE MAE:", mean_absolute_error(y_val, y_base))
print("-" * 30)
#--------------------
param_sets = [
    {"ID": "NN_1", "layers": [256, 128, 64, 32, 16], "regularizer": 0.01},
    {"ID": "NN_2", "layers": [128, 64, 32], "regularizer": 0.01},
    {"ID": "NN_3", "layers": [256, 128], "regularizer": 0.01},
    {"ID": "NN_4", "layers": [512, 256, 128, 64], "regularizer": 0.01}
]

best_val_rmse = float('inf')
best_model = None
best_params = None

for p in param_sets:
    model = Sequential()
    model.add(Input(shape=(X_train.shape[1],)))
    
    # Adds hidden layers based on neurons in the param sets
    for neurons in p["layers"]:
        model.add(Dense(neurons,activation='relu',kernel_regularizer=regularizers.l1(p["regularizer"])))
    model.add(Dense(1)) # Output
    model.compile(loss='mean_squared_error', optimizer='adam')
    #monitor = EarlyStopping(monitor='loss', min_delta=1e-3, patience=5, verbose=1, mode='auto')
    model.summary()
    model.fit(X_train,y_train,verbose=1,epochs=250) #callbacks=[monitor],
    
    print("USING PARAMS:", p)
    
    #Evaluating the model
    y_pred_train = model.predict(X_train) #Training
    y_pred_val   = model.predict(X_val) #Validation

    #for report generation
    train_metrics = calc_metrics(y_train, y_pred_train)
    val_metrics = calc_metrics(y_val, y_pred_val) 
    record_run(nn_runs, "NeuralNetwork", p, train_metrics, val_metrics)
    #--------

    print_metrics("NN_TRAIN", y_train, y_pred_train)
    print_metrics("NN_VAL",   y_val,   y_pred_val)

    # Track best model based on validation RMSE
    if val_metrics["rmse"] < best_val_rmse:
        best_val_rmse = val_metrics["rmse"]
        best_model = model
        best_params = p.copy()
        print(f"Current best model, RMSE: {val_metrics['rmse']/1e6:.2f}M")
    
    print("-" * 30)

if best_params:
    final_model = Sequential()
    final_model.add(Input(shape=(X_trainval.shape[1],)))
    
    for neurons in best_params["layers"]:
        final_model.add(Dense(neurons,activation='relu',kernel_regularizer=regularizers.l1(p["regularizer"])))
    final_model.add(Dense(1))
    final_model.compile(loss='mean_squared_error', optimizer='adam')
    final_model.summary()
    final_model.fit(X_trainval,y_trainval,verbose=1,epochs=250)
    
    write_runs_csv(nn_runs, "Reports/NeuralNetwork/NN_Tuning_Runs.csv") #GENERATE RUNS REPORT

    y_pred_test = best_model.predict(X_test) #predict on x-test set
    print_metrics("NN_TEST", y_test, y_pred_test)
    
    #FINAL RESULTS
    test_results = calc_metrics(y_test, y_pred_test)
    pd.DataFrame([{
        "model": "NeuralNetwork",
        "params": str(best_params),
        "r2_test": test_results["r2"],
        "rmse_test": test_results["rmse"],
        "mae_test": test_results["mae"]
    }]).to_csv("Reports/NeuralNetwork/NN_Test_Result.csv", index=False)
else:
    print("No successful models!")