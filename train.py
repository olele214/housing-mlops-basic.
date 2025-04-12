import pandas as pd

from sklearn.model_selection import train_test_split

from sklearn.linear_model import LinearRegression

import joblib
data = pd.read_csv("Housing.csv")

X = data[["area", "bedrooms", "bathrooms"]]

y = data["price"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = LinearRegression()

model.fit(X_train, y_train)

joblib.dump(model, "model.pkl")

print("Model training complete and saved as model.pkl")
