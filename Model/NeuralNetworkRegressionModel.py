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
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler 
from sklearn.dummy import DummyRegressor

#NEURAL NETWORK REGRESSION MODEL

file_path = 'data/Mojo_budget_update.csv'
data = pd.read_csv(file_path)

#EXPERIMENTATION BLOCKS ---------------------------
# #include distributor as a frequency
# data['distributor'] = data['distributor'].fillna('Unknown')
# data['distributor_freq'] = data['distributor'].map(data['distributor'].value_counts()).fillna(0)

# #convert dates
# data['release_date'] = pd.to_datetime(data['release_date'], format='%d/%m/%Y', errors='coerce')
# data['release_year'] = data['release_date'].dt.year
# data['release_month'] = data['release_date'].dt.month
#EXPERIMENTATION---------------------------

runtime_text = data['run_time'].fillna('')
hours = runtime_text.str.extract(r'(\d+)\s*hr', expand=False).fillna(0).astype(int)
minutes = runtime_text.str.extract(r'(\d+)\s*min', expand=False).fillna(0).astype(int)
data['run_time_minutes'] = hours * 60 + minutes

#DATA CLEANUP / ENCODING (same as linear regression model)
data = data.drop(columns=[
            'movie_id', 'title', 'trivia', 'html',
            'release_date', 'run_time',
            'distributor', 'director', 'writer', 'producer',
            'composer', 'cinematographer',
            'main_actor_1', 'main_actor_2', 'main_actor_3', 'main_actor_4'
        ])

numeric_columns = data.select_dtypes(include=['number']).columns
data[numeric_columns] = data[numeric_columns].fillna(data[numeric_columns].mean())

data['mpaa'] = data['mpaa'].fillna('Unknown')
data = pd.get_dummies(data, columns=['genre_1', 'genre_2', 'genre_3', 'genre_4', 'mpaa'], drop_first=True)

#SPLITTING DATA
y = data["worldwide"] #Target
X = data.drop(columns=['worldwide', 'domestic', 'international']) #Features
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

#Creating model
sc = StandardScaler()
sc.fit(X_train)
X_train= sc.transform(X_train)
X_test = sc.transform(X_test)

input_dim = X.shape[1]
print(f"Input dimension: {input_dim}")

model = Sequential()
model.add(Input(shape=(input_dim,)))
model.add(Dense(256,activation='relu',kernel_regularizer=regularizers.l1(0.01))) # Hidden 1, increased to highest reached 35.64%
model.add(Dense(128, activation='relu',kernel_regularizer=regularizers.l1(0.01))) # Hidden 2, added for actual improvement over mean
model.add(Dense(64, activation='relu',kernel_regularizer=regularizers.l1(0.01))) # Hidden 3, increased to 35%
#model.add(Dropout(0.01))
model.add(Dense(32, activation='relu',kernel_regularizer=regularizers.l1(0.01))) # Hidden 4
model.add(Dense(16,activation='relu',kernel_regularizer=regularizers.l1(0.01))) #Hidden 5

# regularizers: ,kernel_regularizer=regularizers.l2(0.01)

model.add(Dense(1)) # Output
model.compile(loss='mean_squared_error', optimizer='adam')
#monitor = EarlyStopping(monitor='loss', min_delta=1e-3, patience=5, verbose=1, mode='auto')
model.summary()
model.fit(X_train,y_train,verbose=2,epochs=250) #callbacks=[monitor],

#With test data
pred = model.predict(X_test)
score = np.sqrt(metrics.mean_squared_error(pred,y_test))
print(f"Final score (RMSE): {score}")

mean = DummyRegressor(strategy="mean")
mean.fit(X_train, y_train)
y_mean = mean.predict(X_test)
baseline_rmse = np.sqrt(metrics.mean_squared_error(y_test, y_mean))
print(f"Baseline RMSE: {baseline_rmse}")
print(f"Model RMSE: {score}")
print(f"Improvement: {(baseline_rmse - score)/baseline_rmse*100:.2f}%")

# #path to where the file will be saved
# save_path = "save/"

# # save neural network structure to JSON (no weights)
# model_json = model.to_json()
# with open(os.path.join(save_path,"NeuralNetworkRegressor.json"), "w") as json_file:
#     json_file.write(model_json)

# # save entire network to KERAS (save everything, suggested)
# model.save(os.path.join(save_path,"NeuralNetworkRegressor.keras"))

# # code for reloading
# model2 = load_model(os.path.join(save_path,"NeuralNetworkRegressor.keras"))
# pred = model2.predict(X_test)
# score = np.sqrt(metrics.mean_squared_error(pred,y_test))
# print(f"Final score (RMSE): {score}")