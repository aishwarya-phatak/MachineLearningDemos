import pandas as pd

#/Users/vishaljagtap/PycharmProjects/MachineLearningDemos/data/house_price_prediction.csv

#house_price_prediction_dataset = pd.read_csv("/Users/vishaljagtap/PycharmProjects/MachineLearningDemos/data/house_price_prediction.csv")

dataset_1 = pd.read_csv("data/house_price_prediction.csv")
print(dataset_1)
print(dataset_1.keys())
print(dataset_1.describe())