import joblib

import pandas as pd

model = joblib.load("model.pkl")

test_data = pd.DataFrame({"area": [7500], "bedrooms": [3], "bathrooms": [2]})

prediction = model.predict(test_data)

print(f"Predicted price: {prediction[0]}")
