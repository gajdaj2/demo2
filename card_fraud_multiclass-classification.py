import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt



# 1. Generate a Simulated Dataset
np.random.seed(42)

num_samples = 1000
data = {
    "transaction_amount": np.random.exponential(scale=500, size=num_samples),
    "transaction_frequency": np.random.randint(1, 20, size=num_samples),
    "international": np.random.choice([0, 1], size=num_samples, p=[0.8, 0.2]),
    "merchant_type": np.random.choice([0, 1, 2], size=num_samples, p=[0.4, 0.3, 0.3]),  # 0: Retail, 1: Online, 2: Other
    "fraud_category": np.random.choice(
        [0, 1, 2],  # 0: No Fraud, 1: Suspicious, 2: Confirmed Fraud
        size=num_samples,
        p=[0.7, 0.2, 0.1],
    ),
}

df = pd.DataFrame(data)

# 2. Split Features and Target
X = df.drop("fraud_category", axis=1)  # Features
y = df["fraud_category"]  # Target

# Split into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 3. Preprocess the Data
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 4. Train the Random Forest Classifier
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train_scaled, y_train)

# 5. Evaluate the Model
y_pred = model.predict(X_test)

# Classification Report
print("Classification Report:")
print(classification_report(y_test, y_pred, target_names=["No Fraud", "Suspicious", "Confirmed Fraud"]))

# Confusion Matrix
conf_matrix = confusion_matrix(y_test, y_pred)

plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues", xticklabels=["No Fraud", "Suspicious", "Confirmed Fraud"],
            yticklabels=["No Fraud", "Suspicious", "Confirmed Fraud"])
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()


new_data = pd.DataFrame({
    "transaction_amount": [15000, 200],
    "transaction_frequency": [1, 20],
    "international": [1, 0],
    "merchant_type": [0, 2],
})

new_data_scaled = scaler.transform(new_data)
predictions = model.predict(new_data_scaled)
print("\nPredictions for new data:")
print(predictions)