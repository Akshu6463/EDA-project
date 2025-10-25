import numpy as np
from sklearn.ensemble import RandomForestClassifier
import joblib

print("Starting model training...")

# Create mock data
X_train = np.random.rand(100, 5)
y_train = (X_train[:, 0] > 0.5).astype(int)

model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

print("Model training complete.")

joblib.dump(model, 'model.pkl')
print("Model saved to model.pkl")
