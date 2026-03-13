import os
import pickle

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split

# ── Paths ──────────────────────────────────────────────────────────────────────
BASE_DIR  = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, "..", "WineQT.csv")
MODEL_PATH = os.path.join(BASE_DIR, "model.pkl")

# ── Load & clean data ──────────────────────────────────────────────────────────
df = pd.read_csv(DATA_PATH)
df.drop("Id", axis=1, inplace=True)
df = df[~df["quality"].isin([3, 4, 8])]

X = df.drop("quality", axis=1)
y = df["quality"]

# ── Train / test split ─────────────────────────────────────────────────────────
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# ── Train model ────────────────────────────────────────────────────────────────
rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)

# ── Evaluate ───────────────────────────────────────────────────────────────────
y_pred = rf.predict(X_test)
print(classification_report(y_test, y_pred))
print(confusion_matrix(y_test, y_pred))

feat_df = pd.DataFrame(
    {"feature": X.columns, "importance": rf.feature_importances_}
).sort_values(by="importance", ascending=False)
print(feat_df)

# ── Quick smoke-test prediction ────────────────────────────────────────────────
sample = pd.DataFrame([{
    "fixed acidity": 7.4, "volatile acidity": 0.5, "citric acid": 0.3,
    "residual sugar": 2.0, "chlorides": 0.08, "free sulfur dioxide": 15.0,
    "total sulfur dioxide": 100.0, "density": 0.995,
    "pH": 3.2, "sulphates": 0.6, "alcohol": 10.5,
}])
print("Predicted quality:", rf.predict(sample)[0])

# ── Save model (before plt.show() so it never blocks saving) ───────────────────
with open(MODEL_PATH, "wb") as f:
    pickle.dump(rf, f)
print("Model saved to", MODEL_PATH)

# ── Plot quality distribution ──────────────────────────────────────────────────
df["quality"].value_counts().sort_index().plot(kind="bar")
plt.xlabel("Quality")
plt.ylabel("Count")
plt.title("Distribution of Wine Quality")
plt.tight_layout()
plt.show()