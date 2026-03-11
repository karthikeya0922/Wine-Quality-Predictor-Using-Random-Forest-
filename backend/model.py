import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv(r"C:\Users\karth\OneDrive\Desktop\my code\ml project\WineQT.csv")

df.drop("Id", axis=1, inplace=True)
df = df[~df["quality"].isin([3,4,8])]

X = df.drop("quality", axis=1)
y = df["quality"]

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

from sklearn.ensemble import RandomForestClassifier
rf = RandomForestClassifier(n_estimators=100, random_state=42)

rf.fit(X_train, y_train)

y_pred = rf.predict(X_test)

from sklearn.metrics import classification_report, confusion_matrix
print(classification_report(y_test, y_pred))
print(confusion_matrix(y_test, y_pred))

feat_df = pd.DataFrame({
    "feature": X.columns,
    "importance": rf.feature_importances_
}).sort_values(by="importance", ascending=False)

print(feat_df)

new_wine = pd.DataFrame({
    "fixed acidity": [7.4],
    "volatile acidity": [0.5],
    "citric acid": [0.3],
    "residual sugar": [2.0],
    "chlorides": [0.08],
    "free sulfur dioxide": [15.0],
    "total sulfur dioxide": [100.0],
    "density": [0.995],
    "pH": [3.2],
    "sulphates": [0.6],
    "alcohol": [10.5]
})

prediction = rf.predict(new_wine)
print("Predicted quality:", prediction[0])

df["quality"].value_counts().sort_index().plot(kind="bar")

plt.xlabel("Quality")
plt.ylabel("Count")
plt.title("Distribution of Wine Quality")
plt.show()

import pickle
pickle.dump(rf, open("model.pkl", "wb"))