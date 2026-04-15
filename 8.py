import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler

# Load dataset
df = pd.read_csv("wine.csv")

# Features and target
X = df.drop("Class", axis=1)
y = df["Class"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model1 = LogisticRegression(max_iter=10000, random_state=42)
model1.fit(X_train, y_train)
y_pred1 = model1.predict(X_test)
acc1 = accuracy_score(y_test, y_pred1)

# -----------------------------
# 2. WITH STANDARDIZATION
# -----------------------------
scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

model2 = LogisticRegression(max_iter=50, random_state=42)
model2.fit(X_train_scaled, y_train)
y_pred2 = model2.predict(X_test_scaled)
acc2 = accuracy_score(y_test, y_pred2)

print("Accuracy WITHOUT Scaling:", acc1)
print("Accuracy WITH Scaling:", acc2)