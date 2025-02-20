import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OrdinalEncoder, StandardScaler
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
import skops.io as sio

# ------------------------
# 1. Load and Shuffle Data
# ------------------------
drug_df = pd.read_csv("Data/drug200.csv").sample(frac=1, random_state=125)

# ------------------------
# 2. Split Data into Features and Target
# ------------------------
X = drug_df.drop("Drug", axis=1).values
y = drug_df["Drug"].values

# Split into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=125)

# ------------------------
# 3. Define Preprocessing and Model Pipeline
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
# 4. Model Evaluation
# ------------------------
predictions = pipe.predict(X_test)
accuracy = accuracy_score(y_test, predictions)
f1 = f1_score(y_test, predictions, average="macro")

# Print results
print(f"Accuracy: {round(accuracy * 100, 2)}% | F1 Score: {round(f1, 2)}")

# Save metrics to file
with open("Results/metrics.txt", "w") as outfile:
    outfile.write(f"Accuracy = {round(accuracy, 2)}, F1 Score = {round(f1, 2)}\n")

# ------------------------
# 5. Generate and Save Confusion Matrix
# ------------------------
cm = confusion_matrix(y_test, predictions, labels=pipe.classes_)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=pipe.classes_)
disp.plot()
plt.savefig("Results/model_results.png", dpi=120)

# ------------------------
# 6. Save and Load Model Securely
# ------------------------
sio.dump(pipe, "Model/drug_pipeline.skops")

# Load model with trusted types
with open("Model/drug_pipeline.skops", "rb") as f:
    model = sio.load(f, trusted=["numpy.dtype"])