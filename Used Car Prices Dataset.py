#######################################################################################################################
## This code is based on the Automobile dataset presented in the IBM Data Analysis in Python course            ##
#######################################################################################################################

# import pandas library & the source file
import pandas as pd
import numpy as np

# Read the online file by the URL provides above, and assign it to variable "df"
path = "https://s3-api.us-geo.objectstorage.softlayer.net/cf-courses-data/CognitiveClass/DA0101EN/auto.csv"
df = pd.read_csv(path, header=None)

# show the first 5 rows using dataframe.head() method
print("The first 5 rows of the dataframe")
print(df.head(5))

# create headers list (available at https://archive.ics.uci.edu/ml/datasets/Automobile)

headers = ['symboling', 'normalized-losses', 'make', 'fuel-type', 'aspiration', 'num-of-doors', 'body-style',
           'drive-wheels','engine-location', 'wheel-base', 'length', 'width', 'height', 'curb-weight', 'engine-type',
           'num-of-cylinders', 'engine-size', 'fuel-system', 'bore', 'stroke', 'compression-ratio', 'horsepower',
           'peak-rpm', 'city-mpg', 'highway-mpg', 'price']
df.columns = headers
print(df.head(10))


# data wrangling
# we need to replace the "?" symbol with NaN so the dropna() can remove the missing values
df.replace('?', np.nan, inplace=True)

# identify missing values

# The missing values are converted to default. We use the following functions to identify these missing values. There are two methods to detect missing data:
#
# .isnull()
# .notnull()
# The output is a boolean value indicating whether the value that is passed into the argument is in fact missing data.

missing_data = df.isnull()
print(missing_data.head(5))

# Count missing values in each column
# Using a for loop, we can quickly figure out the number of missing values
# in each column. As mentioned above, "True" represents a missing value,
# "False" means the value is present in the dataset.
# In the body of the for loop the method ".value_counts()" counts the number of "True" values.

for column in missing_data.columns.values.tolist():
    print(column)
    print(missing_data[column].value_counts())
    print("")

# For each of the columns that are missing data, we will adopt a particular
# way of dealing with missing data.

# Replace by mean:

avg_norm_loss = df['normalized-losses'].astype('float').mean(axis=0)
df['normalized-losses'].replace(np.nan, avg_norm_loss, inplace=True)

avg_bore=df['bore'].astype('float').mean(axis=0)
df['bore'].replace(np.nan, avg_bore, inplace=True)

avg_stroke = df['stroke'].astype(float).mean(axis=0)
df['stroke'].replace(np.nan, avg_stroke, inplace=True)

avg_horsepower = df['horsepower'].astype('float').mean(axis=0)
df['horsepower'].replace(np.nan, avg_horsepower, inplace=True)

avg_peakrpm=df['peak-rpm'].astype('float').mean(axis=0)
df['peak-rpm'].replace(np.nan, avg_peakrpm, inplace=True)

# For categorical variables, we can replace the missing values with the mode:

# Find out the mode of the number of doors variable
df['num-of-doors'].value_counts()

# or

df['num-of-doors'].value_counts().idxmax()

# replace the missing 'num-of-doors' values by the most frequent
df['num-of-doors'].replace(np.nan, 'four', inplace=True)

# Correct data format

# Convert the columns to the correct data types

print(df.dtypes)

df[['bore', 'stroke']] = df[['bore', 'stroke']].astype('float')
df[['normalized-losses', 'horsepower']] = df[['normalized-losses',  'horsepower']].astype('int')
df[['price']] = df[['price']].astype('float')
df[['peak-rpm']] = df[['peak-rpm']].astype('float')

# Since we will make predictions based on price, we must drop the
# rows with empty values for the price column:

df = df.dropna(subset=['price'], axis=0)

# Check again whether we have empty values:

print("########After data munging:")

missing_data = df.isnull()
for column in missing_data.columns.values.tolist():
    print(column)
    print(missing_data[column].value_counts())
    print("")

# Data standardization & normalization

# Transform mpg to L/100km (if needed uncomment)
#df['city-L/100km'] = 235/df['city-mpg']
#df['highway-mpg'] = 235/df['highway-mpg']

# replace (original value) by (original value)/(maximum value)
# new values will be in the 0-1 range
df['length'] = df['length']/df['length'].max()
df['width'] = df['width']/df['width'].max()
df['height'] = df['height']/df['height'].max()

# Binning the horsepower

bins = np.linspace(min(df["horsepower"]), max(df["horsepower"]), 4)
group_names = ['Low', 'Medium', 'High']
df['horsepower-binned'] = pd.cut(df['horsepower'], bins, labels=group_names, include_lowest=True)

# Indicator variable for fuel types & aspiration

dummy_variable_1 = pd.get_dummies(df["fuel-type"])
dummy_variable_1.rename(columns={'gas': 'fuel-type-gas', 'diesel': 'fuel-type-diesel'}, inplace=True)
dummy_variable_2 = pd.get_dummies(df['aspiration'])


df = pd.concat([df, dummy_variable_1], axis=1)
df = pd.concat([df, dummy_variable_2], axis=1)

# delete unneeded columns

df = df.drop(columns=['fuel-type'])

# save a clean version

df.to_csv('clean_df.csv')

# Exploratory data analysis

import matplotlib.pyplot as plt
import seaborn as sns

# list the data types for each column
print(df.dtypes)

# for example, we can calculate the correlation
# between variables of type "int64" or "float64"
# using the method "corr

print(df.corr())

# Positive linear relationship
df[['engine-size', 'price']].corr()
sns.regplot(x='engine-size', y='price', data=df)
plt.ylim(0,)
plt.show()


df[['highway-mpg', 'price']].corr()
sns.regplot(x='highway-mpg', y='price', data=df)
plt.show()

# Weak relationship
df[['peak-rpm', 'price']].corr()
sns.regplot(x='peak-rpm', y='price', data=df)
plt.show()

# Categorical variables relationship with price

sns.boxplot(x='body-style', y='price', data=df)
plt.show()

sns.boxplot(x='engine-location', y='price', data=df)
plt.show()

sns.boxplot(x='drive-wheels', y="price", data=df)
plt.show()


# Descriptive statistics
print(df.describe())
print(df.describe(include=['object']))

# Example of value counts
drive_wheels_counts = df['drive-wheels'].value_counts().to_frame()
drive_wheels_counts.rename(columns={'drive-wheels': 'value_counts'}, inplace=True)
drive_wheels_counts.index.name = 'drive-wheels'
print(drive_wheels_counts)

# engine-location as variable
engine_loc_counts = df['engine-location'].value_counts().to_frame()
engine_loc_counts.rename(columns={'engine-location': 'value_counts'}, inplace=True)
engine_loc_counts.index.name = 'engine-location'

# grouping
# let's group by the variable "drive-wheels".

# We see that there are 3 different categories of drive wheels.
# We can then calculate the average price for each of the different categories of data.

print(df['drive-wheels'].unique())
df_group_one = df[['drive-wheels', 'body-style', 'price']]
df_group_one = df_group_one.groupby(['drive-wheels'], as_index=False).mean()
print(df_group_one)

# grouping results multiple variable
df_gptest = df[['drive-wheels', 'body-style', 'price']]
grouped_test1 = df_gptest.groupby(['drive-wheels', 'body-style'],as_index=False).mean()
print(grouped_test1)

# Pivoting
grouped_pivot = grouped_test1.pivot(index='drive-wheels',columns='body-style')
grouped_pivot = grouped_pivot.fillna(0) # fill missing values with 0
print(grouped_pivot)

# heatmap using the grouped results
fig, ax = plt.subplots()
im = ax.pcolor(grouped_pivot, cmap='RdBu')

# label names

row_labels = grouped_pivot.columns.levels[1]
col_labels = grouped_pivot.index

# move ticks and labels to the center

ax.set_xticks(np.arange(grouped_pivot.shape[1]) + 0.5, minor=False)
ax.set_yticks(np.arange(grouped_pivot.shape[0]) + 0.5, minor=False)

# insert labels

ax.set_xticklabels(row_labels, minor=False)
ax.set_yticklabels(col_labels, minor=False)

# rotate label if too long

plt.xticks(rotation=90)

fig.colorbar(im)
plt.show()

# Correlation and p-value using stats package

from scipy import stats

pearson_coef, p_value = stats.pearsonr(df['wheel-base'], df['price'])
print('The Pearson Correlation Coefficient is', pearson_coef, ' with a P-value of P =', p_value)

pearson_coef, p_value = stats.pearsonr(df['horsepower'], df['price'])
print('The Pearson Correlation Coefficient is', pearson_coef, ' with a P-value of P = ', p_value)

pearson_coef, p_value = stats.pearsonr(df['length'], df['price'])
print('The Pearson Correlation Coefficient is', pearson_coef, ' with a P-value of P = ', p_value)

pearson_coef, p_value = stats.pearsonr(df['width'], df['price'])
print('The Pearson Correlation Coefficient is', pearson_coef, ' with a P-value of P =', p_value) 

pearson_coef, p_value = stats.pearsonr(df['curb-weight'], df['price'])
print('The Pearson Correlation Coefficient is', pearson_coef, ' with a P-value of P =', p_value)

pearson_coef, p_value = stats.pearsonr(df['engine-size'], df['price'])
print('The Pearson Correlation Coefficient is', pearson_coef, ' with a P-value of P =', p_value)

pearson_coef, p_value = stats.pearsonr(df['city-mpg'], df['price'])
print('The Pearson Correlation Coefficient is', pearson_coef, ' with a P-value of P =', p_value)

pearson_coef, p_value = stats.pearsonr(df['highway-mpg'], df['price'])
print('The Pearson Correlation Coefficient is', pearson_coef, ' with a P-value of P =', p_value)

# ANOVA tests

grouped_test2 = df_gptest[['drive-wheels', 'price']].groupby(['drive-wheels'])

# all 3
f_val, p_val = stats.f_oneway(grouped_test2.get_group('fwd')['price'], grouped_test2.get_group('rwd')['price'],
                              grouped_test2.get_group('4wd')['price'])
print('ANOVA results: F=', f_val, ', P =', p_val)

# 2 by 2

f_val, p_val = stats.f_oneway(grouped_test2.get_group('fwd')['price'], grouped_test2.get_group('rwd')['price'])

print('ANOVA results: F=', f_val, ', P =', p_val)

f_val, p_val = stats.f_oneway(grouped_test2.get_group('4wd')['price'], grouped_test2.get_group('rwd')['price'])

print('ANOVA results: F=', f_val, ', P =', p_val)

f_val, p_val = stats.f_oneway(grouped_test2.get_group('4wd')['price'], grouped_test2.get_group('fwd')['price'])

print('ANOVA results: F=', f_val, ', P =', p_val)

'''

Conclusion: Important Variables
We now have a better idea of what our data looks like and which variables are important to take into account when predicting the car price. We have narrowed it down to the following variables:

Continuous numerical variables:

Length
Width
Curb-weight
Engine-size
Horsepower
City-mpg
highway-mpg
Wheel-base
Bore


Categorical variables:

Drive-wheels

'''

# Predictive Analytics on the automobile set

# Simple linear regression - price as a function of L/100km

from sklearn.linear_model import LinearRegression
lm = LinearRegression()

# Define features & labels set

X = df[['highway-mpg']]
Y = df['price']

# Fit the regressor

lm.fit(X, Y)

# Predict

Yhat = lm.predict(X)

# Check results of the regression and the parameters of the regression line
print(Yhat[0:5])
print(lm.coef_)
print(lm.intercept_)

# Visualize the regressors

width = 12
height = 10
plt.figure(figsize=(width, height))
sns.regplot(x='highway-mpg', y='price', data=df)
plt.ylim(0,)
plt.show()

# Show residuals

width = 12
height = 10
plt.figure(figsize=(width, height))
sns.residplot(df['highway-mpg'], df['price'])
plt.show()


# Multiple Linear Regression
lm1 = LinearRegression()
X1 = df[['horsepower', 'curb-weight', 'engine-size', 'highway-mpg']]
Y1 = df['price']

# Fit the regressor

lm1.fit(X1, Y1)

# Predict

Yhat1 = lm1.predict(X1)

print(Yhat1[0:5])
print(lm1.coef_)
print(lm1.intercept_)

# Visualize the distribution of the predictions vs. actual values

plt.figure(figsize=(width, height))


ax1 = sns.distplot(df['price'], hist=False, color="r", label="Actual Value")
sns.distplot(Yhat1, hist=False, color="b", label="Fitted Values", ax=ax1)


plt.title('Actual vs Fitted Values for Price')
plt.xlabel('Price (in dollars)')
plt.ylabel('Proportion of Cars')

plt.show()
plt.close()

# Polynomial Regression

# Plotting function

def PlotPolly(model, independent_variable, dependent_variabble, Name):
    x_new = np.linspace(15, 55, 100)
    y_new = model(x_new)

    plt.plot(independent_variable, dependent_variabble, '.', x_new, y_new, '-')
    plt.title('Polynomial Fit with Matplotlib for Price ~ Highway MPG')
    ax = plt.gca()
    ax.set_facecolor((0.898, 0.898, 0.898))
    fig = plt.gcf()
    plt.xlabel(Name)
    plt.ylabel('Price of Cars')

    plt.show()
    plt.close()

x = df['highway-mpg']
y = df['price']

# Here we use a polynomial of the 3rd order (cubic)
f = np.polyfit(x, y, 3)
p = np.poly1d(f)
PlotPolly(p, x, y, 'highway-mpg')


# Polynomial transform on multiple features

from sklearn.preprocessing import PolynomialFeatures
pr = PolynomialFeatures(degree=2)
X1_pr = pr.fit_transform(X1)

# Check the shape of the poly transformed dataset:
print(X1_pr.shape, X1.shape)

# Use pipelines for simplifying the whole process (scale, poly transform, model fit)

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

# define the pipeline parameters
Input = [('scale', StandardScaler()), ('polynomial', PolynomialFeatures(include_bias=False)),
       ('model',LinearRegression())]

# Define and fit the pipe objects
pipe = Pipeline(Input)
pipe.fit(X1, Y1)

# Predict
YhatPipe = pipe.predict(X1)
print(YhatPipe[0:3])


# Testing the performance of the models