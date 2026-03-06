import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

#/Users/vishaljagtap/PycharmProjects/MachineLearningDemos/dataset/house_price_prediction.csv

#house_price_prediction_dataset = pd.read_csv("/Users/vishaljagtap/PycharmProjects/MachineLearningDemos/dataset/house_price_prediction.csv")

dataset_1 = pd.read_csv("../dataset/raw/house_price_prediction.csv")
print(dataset_1)
print("-----------------------")
print(dataset_1.keys())
print("-----------------------")
print(dataset_1.describe())
print("-----------------------")
print(dataset_1.info())
print("-----------------------")
print(dataset_1.head())                 #first 5 rows by head method

print("---------Cleaning of data--------------")
print(dataset_1.isnull())           #returns dataframe with boolean true and false values
print(dataset_1.isnull().sum())     #returns dataframe with sum of all null values in each column

print('-------drop values----------')
print(dataset_1.dropna())           #drop the rows where value is missing

print("-----------Encoding-------------")
dataset_with_dummies = pd.get_dummies(dataset_1)            #encoding for categorization
print(dataset_with_dummies)

x = dataset_with_dummies.drop(columns=['Price'])            #drop the columns which you want on the another axes
print(x)
y = dataset_with_dummies['Price']
print(y)

print("--------------Splitting of Dataset For Training and testing purpose------------------")
#scikit learn usage for splitting of dataset
x_train, x_test, y_train, y_test = train_test_split(x,
                                                    y,
                                                    test_size=0.2,
                                                    train_size=0.8,
                                                    random_state=42)
print(x_train)
print(x_test)
print(y_train)
print(y_test)

#Train Random Forest Regression Model
model = RandomForestRegressor(random_state=42, n_estimators=100)
model.fit(x_train, y_train)

#predictions
y_prediction = model.predict(x_test)
print(y_prediction)

#evaluate - MAE, RMSE, R2 Score
mae = mean_absolute_error(y_test, y_prediction)
mse = mean_squared_error(y_test, y_prediction)
result_rms = np.sqrt(mse)
r2_result = r2_score(y_test, y_prediction)
print(mae)
print(result_rms)
print(r2_result)

#matplotlib is a library or framework
plt.figure(figsize=(5,5))
plt.scatter(y_test, y_prediction)
plt.xlabel("Actual Price")
plt.ylabel("Predicted Price")
plt.title("House price prediction")
plt.show()