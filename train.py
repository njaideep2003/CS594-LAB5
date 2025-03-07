import joblib
import os
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

# Load dataset
iris = load_iris()
X_train, X_test, y_train, y_test = train_test_split(iris.data, iris.target, test_size=0.2, random_state=42)

# Train model
model = RandomForestClassifier(n_estimators=100)
model.fit(X_train, y_train)

# ✅ Ensure directory exists inside the shared volume
model_dir = "/app/models"
os.makedirs(model_dir, exist_ok=True)

# ✅ Save model using `protocol=4` for compatibility
model_path = os.path.join(model_dir, "iris_model.pkl")
joblib.dump(model, model_path, compress=3, protocol=4)
print(f"✅ Model saved at {model_path}")
