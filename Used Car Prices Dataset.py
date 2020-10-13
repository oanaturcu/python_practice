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

headers = ["symboling", "normalized-losses", "make", "fuel-type", "aspiration", "num-of-doors", "body-style",
           "drive-wheels","engine-location", "wheel-base", "length", "width", "height", "curb-weight", "engine-type",
           "num-of-cylinders", "engine-size", "fuel-system", "bore", "stroke", "compression-ratio", "horsepower",
           "peak-rpm", "city-mpg", "highway-mpg", "price"]
df.columns = headers
print(df.head(10))


# data wrangling
# we need to replace the "?" symbol with NaN so the dropna() can remove the missing values
df.replace("?", np.nan, inplace=True)

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

