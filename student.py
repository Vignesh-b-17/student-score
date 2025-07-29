import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

# Load data
df = pd.read_csv('enhanced_student_data.csv')

# Features and target
X = df.drop('Final_Score', axis=1)
y = df['Final_Score']

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)

# Train model
model = LinearRegression()
model.fit(X_train, y_train)

# Predict and evaluate
y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
print("Mean Squared Error:", round(mse, 2))

# Predict new student's score
def predict_score():
    h = float(input("Hours Studied: "))
    a = float(input("Attendance Percentage: "))
    s = float(input("Sleep Hours: "))
    i = float(input("Internet Usage (hrs): "))
    p = float(input("Past Grade (%): "))
    e = int(input("Extra Curricular (1=yes, 0=no): "))

    new_pred = model.predict([[h, a, s, i, p, e]])
    print(f" Predicted Final Score: {new_pred[0]:.2f}")

predict_score()