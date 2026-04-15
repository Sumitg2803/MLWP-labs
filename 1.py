# Import required libraries 
import pandas as pd 
from sklearn.model_selection import train_test_split 
from sklearn.neighbors import KNeighborsClassifier 
from sklearn.metrics import classification_report 
 
# Step 1: Load dataset 
df = pd.read_csv("Iris.csv") 
 
# Step 2: Separate features (X) and target (y) 
X = df.iloc[:, :-1]   # All columns except last 
y = df.iloc[:, -1]    # Last column (species) 
 
# Step 3: Split dataset into training and testing sets 
X_train, X_test, y_train, y_test = train_test_split( 
    X, y, test_size=0.2, random_state=42 
) 
 
# Step 4: Create and train the model 
model = KNeighborsClassifier(n_neighbors=3) 
model.fit(X_train, y_train) 
 
# Step 5: Make predictions 
y_pred = model.predict(X_test) 
 
# Step 6: Evaluate the model 
accuracy = model.score(X_test, y_test) 
print("Accuracy:", accuracy) 
 
print("\nClassification Report:\n") 
print(classification_report(y_test, y_pred)) 