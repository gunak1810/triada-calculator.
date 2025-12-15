import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

# 1. Load your dataset
print("⏳ Loading data...")
df = pd.read_csv("NHANES_2009_2018_Cleaned.csv")

# 2. Select Features (The 5 Variables)
features = ["RIDAGEYR", "RIAGENDR", "BMXBMI", "BMXWT", "LBXGLU"]
target = "IR_LABEL"

# Clean data
df_model = df.dropna(subset=features + [target]).copy()
X = df_model[features]
y = df_model[target]

# 3. Train the Model
print("🤖 Training TRIADA-IR Model...")
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, stratify=y, random_state=42
)

rf = RandomForestClassifier(n_estimators=300, max_depth=6, random_state=42)
rf.fit(X_train, y_train)

# 4. Save the Model to a File
model_filename = "triada_ir_model.pkl"
joblib.dump(rf, model_filename)

print(f"✅ Success! Model saved as '{model_filename}'")
print("You can now move this file to your app folder.")