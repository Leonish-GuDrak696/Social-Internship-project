import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

# Load dataset
df = pd.read_csv("CEAS_08.csv")

# Combine subject and body
X = (
    df["subject"].fillna("").astype(str)
    + " "
    + df["body"].fillna("").astype(str)
)

y = df["label"]

# Convert text to numerical features
vectorizer = TfidfVectorizer(
    stop_words="english",
    max_features=10000
)

X_vec = vectorizer.fit_transform(X)

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X_vec,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

# Train model
model = LogisticRegression(
    max_iter=1000,
    random_state=42
)

model.fit(X_train, y_train)

# Evaluate
y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)

print(f"\nAccuracy: {accuracy:.4f}")
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# Save model and vectorizer
joblib.dump(model, "phishing_model.pkl")
joblib.dump(vectorizer, "vectorizer.pkl")

print("\nModel and vectorizer saved successfully.")
print(model.classes_)