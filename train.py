import pandas as pd
import pickle

from sklearn.linear_model import LinearRegression

data = pd.read_csv("emp.csv")
X = data[["YearsExperience"]]
y = data["Salary"]

model = LinearRegression()

model.fit(X, y)

with open("salary_model.pkl", "wb") as file:
    pickle.dump(model, file)

print("Model trained successfully!")