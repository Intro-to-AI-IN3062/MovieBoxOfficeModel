import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
#SIMPLE LINEAR REGRESSION MODEL

file_path = 'data/Mojo_budget_update.csv'
data = pd.read_csv(file_path)

#Drop irrelevant columns
data = data.drop(columns=['movie_id', 'title', 'trivia', 'html'])

#NULL VALUES
missing_data = data.isnull().sum()
missing_percentage = (missing_data / len(data)) * 100
#use mean of column to fill missing numerical values
numeric_columns = data.select_dtypes(include=['number']).columns
data[numeric_columns] = data[numeric_columns].fillna(data[numeric_columns].mean())

#ENCODING
#One-Hot Encoding
data = pd.get_dummies(data, columns=['genre_1', 'genre_2', 'genre_3', 'genre_4'], drop_first=True) #converts categorical columns into binary columns

#Label Encoding
label_encoder = LabelEncoder()
data['mpaa'] = label_encoder.fit_transform(data['mpaa'].astype(str))

#Splitting Data-----
y = data["worldwide"] #Target

X = data.drop(columns=['worldwide', 'domestic', 'international']) #Features
X = X.drop(columns=[
    'release_date', 'run_time',
    'distributor', 'director', 'writer', 'producer',
    'composer', 'cinematographer',
    'main_actor_1', 'main_actor_2', 'main_actor_3', 'main_actor_4'
])

print("Features: \n", X.head())
print("Target: \n", y.head())
print("Target Types: \n", X.dtypes)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)
print(X_train.shape, X_test.shape)
print(y_train.shape, y_test.shape)

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

print("R^2:", r2)

print(f"R^2 accuracy: {r2 * 100}%")
