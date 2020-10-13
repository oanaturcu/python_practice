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

df.dtypes
df[['bore', 'stroke']] = df[['bore', 'stroke']].astype('float')
df[['normalized-losses']] = df[['normalized-losses']].astype('int')
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

# Data standardization

