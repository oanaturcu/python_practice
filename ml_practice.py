# 1. Classification of weather data usking scikit-learn
# Based on the weather conditions at 9 AM, predict whether
# we will have high humidity at 3 PM (high humidity means over 25)

# Import libraries and data source file

import pandas as pd
import numpy as np
import sklearn as skl
import sklearn.metrics as skm
import sklearn.model_selection as skmd
import sklearn.tree as skt

data = pd.read_csv('./ml_datasets/daily_weather.csv')


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
work_data['high_humidity_label'] = (work_data['relative_humidity_3pm']>25)*1
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

## Set prediction value to "High" or "Low" based on 1 and 0 values

x_test_out.loc[x_test_out['prediction'] == 1, 'predicted_hum_3pm'] = 'High'
x_test_out.loc[x_test_out['prediction'] == 0, 'predicted_hum_3pm'] = 'Low'
del x_test_out['prediction']

## Export the predictions to csv for further operations:

x_test_out.to_csv('./exports/weather_predictions.csv')


