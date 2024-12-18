from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
import pandas as pd
# Sample dataset
data = {
    "size": [1000, 1200, 1500, 800, 950],
    "location_index": [1, 2, 3, 1, 2],
    "price": [200000, 250000, 300000, 150000, 220000],
}
df = pd.DataFrame(data)

# Features and target
X = df[["size", "location_index"]]
y = df["price"]

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Model training
model = LinearRegression()
model.fit(X_train, y_train)

# Prediction and evaluation
print(X_test)
y_pred = model.predict(X_test)
print(y_pred)
print("Mean Absolute Error:", mean_absolute_error(y_test, y_pred))