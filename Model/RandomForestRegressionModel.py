import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.ensemble import RandomForestRegressor
import pandas as pd
from sklearn.model_selection import train_test_split
#RANDOM FOREST REGRESSION MODEL

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
model = RandomForestRegressor(
    n_estimators=200,
    random_state=42,
    n_jobs=1,
)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

r2 = r2_score(y_test, y_pred)
print("Random Forest R^2:", r2)
print(f"Random Forest R^2 accuracy: {r2 * 100}%")

rmse = np.sqrt(mean_squared_error(y_test, y_pred))
print("RFR RMSE: ", rmse)
print("RFR MAE:", mean_absolute_error(y_test, y_pred))

