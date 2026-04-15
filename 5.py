# Import required libraries 
import pandas as pd 
from sklearn.model_selection import train_test_split 
 
 
 
from sklearn.linear_model import LinearRegression 
from sklearn.metrics import mean_absolute_error, mean_squared_error 
 
# Step 1: Load dataset 
df = pd.read_csv("housing.csv") 
 
# Step 2: Handle missing values 
df.fillna(df.mean(numeric_only=True), inplace=True) 
 
# Step 3: Separate features and target 
X = df.drop("Price", axis=1) 
y = df["Price"] 
 
# Step 4: Split dataset 
X_train, X_test, y_train, y_test = train_test_split( 
    X, y, test_size=0.2, random_state=42 
) 
 
# Step 5: Train model 
model = LinearRegression() 
model.fit(X_train, y_train) 
 
# Step 6: Prediction 
y_pred = model.predict(X_test) 
 
# Step 7: Evaluation 
r2 = model.score(X_test, y_test) 
mae = mean_absolute_error(y_test, y_pred) 
mse = mean_squared_error(y_test, y_pred) 
 
print("R2 Score:", r2) 
print("MAE:", mae) 
print("MSE:", mse) 