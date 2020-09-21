### This book is about getting myself familiar with basic ML operations:
# classification, regression, clustering, as well as connecting to SQL databases with Python
# No monetary purpose


# 1. Classification of weather data using scikit-learn

# Based on the weather conditions at 9 AM, predict whether
# we will have high humidity at 3 PM (high humidity means over 25)
# The file **daily_weather.csv** is a comma-separated file that contains weather data.
# This data comes from a weather station located in San Diego, California. The weather station is equipped
# with sensors that capture weather-related measurements such as air temperature, air pressure, and relative humidity.
# Data was collected for a period of three years, from September 2011 to September 2014, to ensure that sufficient data
# for different seasons and weather conditions is captured.
# Dataset kindly provided by UC San Diego as part of the Python for Data Science course.

# Import libraries and data source file

import pandas as pd
import numpy as np
import sklearn as skl
import sklearn.metrics as skm
import sklearn.model_selection as skmd
import sklearn.tree as skt

data = pd.read_csv('./ml_datasets/weather/daily_weather.csv')


# Explore & clean the data

print(data.columns)
print(data.head())
print(data[data.isnull().any(axis=1)])
print("+++++  ", data.shape[0])

## drop rows with empty data, delete unnecessary columns

data = data.dropna()
del data['number']

print("+++++  ", data.shape[0])

# Build the classifier

## Create the prediction label

work_data = data.copy()
work_data['high_humidity_label'] = (work_data['relative_humidity_3pm'] > 25)*1
print(work_data.columns)

## Store the target label in a separate df:

y = work_data[['high_humidity_label']].copy()
print(y)

## Create the features dataset and store in a dataframe called x
## (to keep the y=f(x) analogy)

morning_features = ['air_pressure_9am','air_temp_9am','avg_wind_direction_9am','avg_wind_speed_9am',
        'max_wind_direction_9am','max_wind_speed_9am','rain_accumulation_9am',
        'rain_duration_9am']

x = work_data[morning_features].copy()
print(x.columns)

## Perform test & train split
## 30-70 split on train / test data
## try different random states to maximize accuracy

x_train, x_test, y_train, y_test = skmd.train_test_split(x, y, test_size=0.30,
                                                                            random_state=324)

## Fit on Train set
## Try different max leaf nodes and random state to increase accuracy

humidity_classifier = skt.DecisionTreeClassifier(max_leaf_nodes=20, random_state=0)
humidity_classifier.fit(x_train, y_train)

## Predict on Test set
predictions = humidity_classifier.predict(x_test)

print(skm.accuracy_score(y_true = y_test, y_pred=predictions))

## Merge initial data and prediction in the same dataframe
predictions_df = pd.DataFrame(data=predictions, columns=['prediction'], index=x_test.index.copy())
x_test_out = pd.merge(x_test, predictions_df, how='left', left_index=True, right_index=True)

## Set prediction value to "Hi" or "Low" based on 1 and 0 values

x_test_out.loc[x_test_out['prediction'] == 1, 'predicted_hum_3pm'] = 'Hi'
x_test_out.loc[x_test_out['prediction'] == 0, 'predicted_hum_3pm'] = 'Low'
del x_test_out['prediction']

## Export the predictions to csv for further operations:

x_test_out.to_csv('./exports/weather_predictions.csv')


# 2. Prediction on IRIS dataset (https://www.kaggle.com/uciml/iris).

# The Iris dataset was used in R.A. Fisher's classic 1936 paper,
# The Use of Multiple Measurements in Taxonomic Problems,
# and can also be found on the UCI Machine Learning Repository.

#It includes three iris species with 50 samples each as well as some properties about each flower.
# One flower species is linearly separable from the other two,
# but the other two are not linearly separable from each other.

#The columns in this dataset are:

# Id
# SepalLengthCm
# SepalWidthCm
# PetalLengthCm
# PetalWidthCm
# Species


# The classical ML task on this dataset is to predict the species based on petal and sepal measurement

# import sqlite library & connect to database (create a dataframe)
# import seaborn in order to do some data exploration

import sqlite3
import seaborn as sns
from matplotlib import pyplot

conn = sqlite3.connect('./ml_datasets/iris/database.sqlite')
iris_data = pd.read_sql_query("SELECT * FROM Iris", conn)
print(iris_data.head())

# Perform data exploration & cleanup

print(iris_data.columns)
print("+++++  ", iris_data.shape[0])
print(iris_data.describe())
print(iris_data.info())

sns.distplot(iris_data['SepalLengthCm'], kde=False, bins=60)
pyplot.show()

sns.distplot(iris_data['SepalWidthCm'],kde=False, bins=60)
pyplot.show()

sns.distplot(iris_data['PetalLengthCm'], kde=False, bins=60)
pyplot.show()

sns.distplot(iris_data['PetalWidthCm'], kde=False, bins=60)
pyplot.show()

sns.jointplot('SepalLengthCm', 'SepalWidthCm', iris_data, space=0.4)
pyplot.show()

sns.jointplot('PetalLengthCm', 'PetalWidthCm', iris_data, space=0.4)
pyplot.show()

print(iris_data.corr())

sns.heatmap(iris_data.corr(), annot=True)
pyplot.show()

sns.pairplot(iris_data, hue='Species')
pyplot.show()

# Build x and y

iris_work = iris_data.copy()

y_iris = iris_work[['Species']].copy()
del iris_work['Species']
x_iris = iris_work.copy()
x_iris_train, x_iris_test, y_iris_train, y_iris_test = skmd.train_test_split(x_iris, y_iris, test_size=0.30,
                                                                            random_state=324)

# Apply various classifier models

## Decision tree classifier

iris_classifier = skt.DecisionTreeClassifier(max_leaf_nodes=20, random_state=0)
iris_classifier.fit(x_iris_train, y_iris_train)
predictions_iris = iris_classifier.predict(x_iris_test)
print("@@@@@@ Decision tree accuracy: ", skm.accuracy_score(y_true=y_iris_test, y_pred=predictions_iris))

# The simple decision tree algorithm already predicts flawlessly, but let's try a few more

## Random forest classifier

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix

rf_iris_classifier = RandomForestClassifier(n_estimators=80, verbose=5, random_state=42)
rf_iris_classifier.fit(x_iris_train, y_iris_train)
predictions_iris_rf = rf_iris_classifier.predict(x_iris_test)
print(predictions_iris_rf)
print(confusion_matrix(y_iris_test, predictions_iris_rf))
print("#### RF Accuracy: ", rf_iris_classifier.score(x_iris_test, predictions_iris_rf))

## Logistical Regression

from sklearn.linear_model import LogisticRegression

logr = LogisticRegression()
logr.fit(x_iris_train, y_iris_train)
predictions_iris_logr = logr.predict(x_iris_test)
print(confusion_matrix(y_iris_test, predictions_iris_logr))
print("%%%% Logistical Regression Accuracy: ", logr.score(x_iris_test, predictions_iris_logr))
