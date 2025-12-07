import pandas as pd
from sklearn.preprocessing import LabelEncoder

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

data = data.select_dtypes(include=['number'])

print("Missing data count:\n", missing_data[missing_data > 0])
print("Missing data percentage:\n", missing_percentage[missing_percentage > 0])
print(data.head())
print(data.columns)