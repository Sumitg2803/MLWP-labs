# Import required libraries 
import pandas as pd 
from sklearn.model_selection import train_test_split 
from sklearn.neighbors import KNeighborsClassifier 
from sklearn.metrics import classification_report 
from sklearn.preprocessing import LabelEncoder, StandardScaler 
 
# Step 1: Load dataset 
df = pd.read_csv("Titanic.csv") 
 
# Step 2: Handle missing values 
df['Age'].fillna(df['Age'].mean(), inplace=True) 
 
# Step 3: Drop unnecessary columns 
df.drop(['Cabin', 'Name', 'Ticket'], axis=1, inplace=True) 
 
# Step 4: Encode categorical data 
le = LabelEncoder() 
for col in df.select_dtypes(include='object').columns: 
    df[col] = le.fit_transform(df[col]) 
 
# Step 5: Separate features and target 
X = df.drop("Survived", axis=1) 
y = df["Survived"] 
 
# Step 6: Split dataset 
X_train, X_test, y_train, y_test = train_test_split( 
    X, y, test_size=0.2, random_state=42 
) 
 
# Step 7: Feature scaling 
scaler = StandardScaler() 
X_train = scaler.fit_transform(X_train) 
X_test = scaler.transform(X_test) 
 
# Step 8: Train model 
model = KNeighborsClassifier(n_neighbors=5) 
model.fit(X_train, y_train) 
 
# Step 9: Prediction 
y_pred = model.predict(X_test) 
 
# Step 10: Evaluation 
accuracy = model.score(X_test, y_test) 
print("Accuracy:", accuracy) 
 
print("\nClassification Report:\n") 
print(classification_report(y_test, y_pred)) 