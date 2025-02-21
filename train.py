import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OrdinalEncoder, StandardScaler
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

# Try importing skops, otherwise, exit gracefully
try:
    import skops.io as sio
except ImportError:
    print("⚠️ 'skops' is not installed! Run `pip install skops` before proceeding.")
    exit(1)

# ------------------------
# 1️⃣ Ensure Directories Exist
# ------------------------
os.makedirs("Results", exist_ok=True)
os.makedirs("Model", exist_ok=True)

# ------------------------
# 2️⃣ Load and Shuffle Data
# ------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))  # Get script directory
DATA_PATH = os.path.join(BASE_DIR, "..", "DATA", "drug200.csv")  # Correct dataset path

# Ensure dataset exists
if not os.path.exists(DATA_PATH):
    raise FileNotFoundError(f"❌ Dataset not found at: {DATA_PATH}")

# Load and shuffle data
drug_df = pd.read_csv(DATA_PATH).sample(frac=1, random_state=125)

# ------------------------
# 3️⃣ Split Data into Features and Target
# ------------------------
X = drug_df.drop("Drug", axis=1).values
y = drug_df["Drug"].values

# Split into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=125)

# ------------------------
# 4️⃣ Define Preprocessing and Model Pipeline
# ------------------------
cat_col = [1, 2, 3]  # Categorical feature indices
num_col = [0, 4]  # Numerical feature indices

transform = ColumnTransformer([
    ("encoder", OrdinalEncoder(), cat_col),
    ("num_imputer", SimpleImputer(strategy="median"), num_col),
    ("num_scaler", StandardScaler(), num_col),
])

pipe = Pipeline([
    ("preprocessing", transform),
    ("model", RandomForestClassifier(n_estimators=100, random_state=125)),
])

# Train the model
pipe.fit(X_train, y_train)

# ------------------------
# 5️⃣ Model Evaluation
# ------------------------
predictions = pipe.predict(X_test)
accuracy = accuracy_score(y_test, predictions)
f1 = f1_score(y_test, predictions, average="macro")

# Print results
print(f"✅ Accuracy: {round(accuracy * 100, 2)}% | F1 Score: {round(f1, 2)}")

# Save metrics to file
with open("Results/metrics.txt", "w") as outfile:
    outfile.write(f"Accuracy = {round(accuracy, 2)}, F1 Score = {round(f1, 2)}\n")

# ------------------------
# 6️⃣ Generate and Save Confusion Matrix
# ------------------------
cm = confusion_matrix(y_test, predictions, labels=pipe.classes_)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=pipe.classes_)
disp.plot()
plt.savefig("Results/model_results.png", dpi=120)

# ------------------------
# 7️⃣ Save and Load Model Securely
# ------------------------
MODEL_PATH = os.path.join(BASE_DIR, "..", "Model", "drug_pipeline.skops")
sio.dump(pipe, MODEL_PATH)

# Ensure model file exists
if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"❌ Model file not found at: {MODEL_PATH}")

with open(MODEL_PATH, "rb") as f:
    untrusted_types = sio.get_untrusted_types(f)  # Get untrusted types
    print("🔍 Untrusted types found:", untrusted_types)
    model = sio.load(f, trusted=untrusted_types)

print("✅ Model training and evaluation completed successfully!")
