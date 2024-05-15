import os.path

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from tqdm import tqdm

import joblib

filenames = ['USvideos.csv', 'CAvideos.csv', 'DEvideos.csv', 'GBvideos.csv', 'INvideos.csv', 'RUvideos.csv']  # Add more filenames as needed
dataframes = []
for filename in filenames:
    df = pd.read_csv(os.path.join('training_data', filename), encoding='ISO-8859-1')
    dataframes.append(df)
dataset = pd.concat(dataframes, ignore_index=True)
dataset = dataset.sample(frac=1, random_state=42).reset_index(drop=True)
dataset = dataset[:100000]

categorical_features = ['title', 'channel_title', 'publish_time', 'tags']
categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])

numerical_features = ['category_id']
numerical_transformer = SimpleImputer(strategy='constant', fill_value=0)

preprocessor = ColumnTransformer(
    transformers=[
        ('cat', categorical_transformer, categorical_features),
        ('num', numerical_transformer, numerical_features)
    ])

X = dataset[['title', 'channel_title', 'category_id', 'publish_time', 'tags', 'description']]
y = dataset[['views', 'likes', 'comment_count']]

X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)


model = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('regressor', RandomForestRegressor(n_estimators=100, random_state=42))
])
n_iterations = len(X_train) // model['regressor'].n_estimators

# Train the model with a progress bar
with tqdm(total=model['regressor'].n_estimators) as pbar:
    for i in range(model['regressor'].n_estimators):
        model.fit(X_train[n_iterations*i:n_iterations*(i+1)], y_train[n_iterations*i:n_iterations*(i+1)])
        pbar.update(1)

y_pred = model.predict(X_val)
mse = mean_squared_error(y_val, y_pred)
print('Mean Squared Error:', mse)

joblib.dump(model, 'random_forest_model.pkl')
