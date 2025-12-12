import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import numpy as np

#SIMPLE LINEAR REGRESSION MODEL

file_path = 'data/Mojo_budget_update.csv'
data = pd.read_csv(file_path)

#DATA CLEANUP
data = data.drop(columns=[
            'movie_id', 'title', 'trivia', 'html',
            'release_date', 'run_time',
            'distributor', 'director', 'writer', 'producer',
            'composer', 'cinematographer',
            'main_actor_1', 'main_actor_2', 'main_actor_3', 'main_actor_4'
        ])
#use mean of column to fill missing numerical values
numeric_columns = data.select_dtypes(include=['number']).columns
data[numeric_columns] = data[numeric_columns].fillna(data[numeric_columns].mean())

#ENCODING
data['mpaa'] = data['mpaa'].fillna('Unknown')
data = pd.get_dummies(data, columns=['genre_1', 'genre_2', 'genre_3', 'genre_4', 'mpaa'], drop_first=True) #converts categorical columns into binary columns

# #remove extreme values
# q99 = data['worldwide'].quantile(0.99)
# data = data[data['worldwide'] <= q99].copy()


#Splitting Data-----
y = data["worldwide"] #Target

X = data.drop(columns=['worldwide', 'domestic', 'international']) #Features
# numeric_feature_columns = X.select_dtypes(include=['number']).columns
# X[numeric_feature_columns] = np.log1p(X[numeric_feature_columns])

print("CHECK NA : ", X.isna().sum().sort_values(ascending=False).head(10))

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

#Simple linear regression model
model = LinearRegression()
model.fit(X_train, y_train)

#Predictions
y_predicted = model.predict(X_test)
results = pd.DataFrame({
    'true_worldwide': y_test.values,
    'predicted_worldwide': y_predicted
})
print("Predictions: \n")
print(results.head(20))

#Evaluation
print("Number of coefficients:", len(model.coef_)) #weights
r2 = r2_score(y_test, y_predicted)

print("Linear Regression R^2:", r2)
print(f"Linear Regression R^2 accuracy: {r2 * 100}%")

rmse = np.sqrt(mean_squared_error(y_test, y_predicted))
print("LR RMSE: ", rmse)
print("LR MAE:", mean_absolute_error(y_test, y_predicted))