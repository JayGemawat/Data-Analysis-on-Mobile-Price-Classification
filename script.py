import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
import joblib

# Get the base directory where app2.py is located
base_path = os.path.dirname(os.path.abspath(__file__))

# Set file paths relative to the base directory
train_file_path = os.path.join(base_path, "train.csv")
test_file_path = os.path.join(base_path, "test.csv")
model_file_path = os.path.join(base_path, "model.pkl")

# Load dataset using the relative path
train_data = pd.read_csv(train_file_path)

# Features and target variable (Update 'price_range' to your target column name)
X = train_data.drop(columns=['price_range'])  # Features
y = train_data['price_range']  # Target variable

# Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize and train a Decision Tree Classifier (or any other model)
model = DecisionTreeClassifier(random_state=42)
model.fit(X_train, y_train)

# Evaluate the model
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Model Accuracy: {accuracy * 100:.2f}%")

# Save the trained model using the relative path
joblib.dump(model, model_file_path)
print(f"Model saved as '{model_file_path}'")
