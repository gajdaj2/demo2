import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
from sklearn.pipeline import Pipeline

# 1. Sample Dataset: IT Defect Descriptions
data = {
    "description": [
        "System crashes when submitting the form.",
        "Page loading is extremely slow under heavy traffic.",
        "Error message is unclear and confusing.",
        "Search functionality returns irrelevant results.",
        "System does not save user preferences after logout.",
        "Add dark mode feature as requested by users.",
        "Mobile application crashes when uploading large files.",
        "Button alignment issue on the homepage.",
        "API throws a 500 error intermittently.",
        "Database connection is not established under load."
    ],
    "category": [
        "Critical",  # Critical issue: Crashes
        "Performance",  # Performance issue
        "Usability",  # Usability issue
        "Functionality",  # Functional issue
        "Functionality",  # Functional issue
        "Feature Request",  # New feature request
        "Critical",  # Critical issue: Crashes
        "Usability",  # Usability issue
        "Critical",  # Critical issue: Server error
        "Performance"  # Performance issue
    ]
}

# Convert to a DataFrame
df = pd.DataFrame(data)

# 2. Split Data into Training and Testing Sets
X = df["description"]  # Features (defect descriptions)
y = df["category"]  # Labels (categories)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 3. Build a Text Classification Pipeline
pipeline = Pipeline([
    ("tfidf", TfidfVectorizer(stop_words="english")),  # Convert text to numerical features
    ("classifier", RandomForestClassifier(n_estimators=100, random_state=42))  # Classifier
])

# 4. Train the Model
pipeline.fit(X_train, y_train)

# 5. Make Predictions
y_pred = pipeline.predict(X_test)

# 6. Evaluate the Model
print("Classification Report:")
print(classification_report(y_test, y_pred))
print(f"Accuracy: {accuracy_score(y_test, y_pred):.2f}")

# 7. Example Predictions
new_defects = [
    "System crashes during login when using multiple sessions.",
    "The website is slow when accessed from mobile devices.",
    "Add a feature to export reports in PDF format.",
    "Buttons are misaligned on the checkout page."
]

predictions = pipeline.predict(new_defects)

print("\nPredictions for New Defects:")
for defect, category in zip(new_defects, predictions):
    print(f"Defect: {defect} => Predicted Category: {category}")

# Save the model
import joblib
joblib.dump(pipeline, "defect_classifier.pkl")

from sklearn.utils import estimator_html_repr
html = estimator_html_repr(pipeline)


with open("pipeline_visualization.html", "w") as f:
    f.write(html)

from IPython.core.display import display, HTML
display(HTML(html))