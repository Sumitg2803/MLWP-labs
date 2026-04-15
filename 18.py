import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.feature_selection import RFE


df = pd.read_csv("Titanic.csv")


df = df[["Survived", "Pclass", "Sex", "Age", "Fare"]]


df["Age"] = df["Age"].fillna(df["Age"].median())

df["Sex"] = LabelEncoder().fit_transform(df["Sex"])


X = df.drop("Survived", axis=1)
y = df["Survived"]

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)


model = LogisticRegression(max_iter=1000)

# RFE (select top 2 features)
rfe = RFE(estimator=model, n_features_to_select=2)
rfe.fit(X_train, y_train)

# Selected features
selected_features = X.columns[rfe.support_]

print("Selected Features:", list(selected_features))