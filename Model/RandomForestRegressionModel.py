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

#RANDOM FOREST REGRESSION MODEL
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
#-------------------------------

#Import data
file_path = 'data/Mojo_budget_update.csv'
data = pd.read_csv(file_path)

runtime_text = data['run_time'].fillna('')
hours = runtime_text.str.extract(r'(\d+)\s*hr', expand=False).fillna(0).astype(int)
minutes = runtime_text.str.extract(r'(\d+)\s*min', expand=False).fillna(0).astype(int)
data['run_time_minutes'] = hours * 60 + minutes

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

#PREPROCESSING (updated to use pipeline)
categorical_cols = ['genre_1','genre_2','genre_3','genre_4','mpaa']
numeric_cols = X.select_dtypes(include=['number']).columns.tolist()

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

#BASELINE MODEL (uses pipeline)
baseline = Pipeline([
    ("preprocess", make_preprocess()),
    ("model", DummyRegressor(strategy="mean")),
])
baseline.fit(X_train, y_train)

y_base = baseline.predict(X_val)
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
    ("preprocess", make_preprocess()),
    ("model", RandomForestRegressor(
        n_estimators=p['n_estimators'],
        max_depth=p['max_depth'],
        min_samples_leaf=p["min_samples_leaf"],
        max_features=p["max_features"],
        random_state=42,
        n_jobs=1,
    )),
])
    model.fit(X_train, y_train)
    print("USING PARAMS:", p)

    #Evaluating the model
    y_pred_train = model.predict(X_train) #Training
    y_pred_val   = model.predict(X_val) #Validation

    print_metrics("RF_TRAIN", y_train, y_pred_train)
    print_metrics("RF_VAL",   y_val,   y_pred_val)

    print("-" * 30)

#Final Evaluation of best model
best_p = {"n_estimators": 200, "max_depth": None, "min_samples_leaf": 1, "max_features": "sqrt"}
best_model = Pipeline([
    ("preprocess", make_preprocess()),
    ("model", RandomForestRegressor(
        n_estimators=best_p["n_estimators"],
        max_depth=best_p["max_depth"],
        min_samples_leaf=best_p["min_samples_leaf"],
        max_features=best_p["max_features"],
        random_state=42,
        n_jobs=1,
    )),
])

best_model.fit(X_trainval, y_trainval) #refit model
y_pred_test = best_model.predict(X_test) #predict on x-test set
print_metrics("RF_TEST", y_test, y_pred_test)