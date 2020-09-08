# This is a practice book based on the IMDB small dataset
# available from https://grouplens.org/datasets/movielens/
# Its aim is to familiarize myself with the basics of
# operating with libraries such as pandas, numpy and matplotlib as well as
# documenting my progress.
# No monetary purpose.


# Import the needed libraries here

import pandas as pd
import matplotlib.pyplot as plt


# Load the needed datasets here
ratings = pd.read_csv('imdb_datasets/ratings.csv')
tags = pd.read_csv('imdb_datasets/tags.csv')
movies = pd.read_csv('imdb_datasets/movies.csv')

# Perform basic DS operations:

# 1. Exploratory analysis of the data

# 1.1 Have a look at what each dataset contains
print(ratings.head())
print(tags.head())
print(movies.head())

# 1.2 Perform data quality checks - null values &

print(ratings.isnull().any())
print(tags.isnull().any())
print(movies.isnull().any())

# 1.3 Delete rows with missing tag values from the tags dataframe.
# Note that the initial file is not modified, only the dataframe.

tags = tags.dropna()

# Check to see if the tags dataframe still contains null values

print(tags.isnull().any())

# 1.4 Descriptive statistics.

# Note that this works only on numeric values,
# therefore it will not do anything on # columns with string values.
# It will still run but it will leave out those columns.
# Furthermore, the only relevant statistics are on ratings,
# since movie ID's and user ID's are identifiers.
# Therefore we will perform the describe method
# only on the 'rating' column of the ratings dataframe.


print(ratings['rating'].describe())

# The describe method allows us to perform a sanity check on the data as well.
# We expect the values in the ratings  column to be numerical,
# greater than 0 and less than or equal to 5.

# 1.5 Exploratory Visualisation with matplotlib.
# Since this is exploratory, no attention will be given to formatting.

# 1.5.1 Exploring rating distribution:

x = ratings['rating']

plt.boxplot(x)
plt.show()

plt.hist(x)
plt.show()

# 1.5.2 Exploring tags: getting the counts for each tag and
# displaying and plotting top 10

tag_counts = tags['tag'].value_counts()
print(tag_counts[:10])
tag_counts[:10].plot(kind='bar')
plt.show()

# 2. Creating filtered datasets for various criteria

# 2.1 Movies tagged as various genres
is_comedy = movies['genres'].str.contains('Comedy')
is_drama = movies['genres'].str.contains('Drama')
is_thriller = movies['genres'].str.contains('Thriller')
is_fantasy = movies['genres'].str.contains('Fantasy')
is_romance = movies['genres'].str.contains('Romance')
is_sf = movies['genres'].str.contains('Sci-Fi')
is_action = movies['genres'].str.contains('Action')
is_animation = movies['genres'].str.contains('Animation')
is_adventure = movies['genres'].str.contains('Adventure')


# 2.2 Create various subsets (1 filter+) and plot top n (10 in this case)
# Formatting is used for axes labels etc

comedies = movies[is_comedy]
comedies_count = comedies['genres'].value_counts()
print(comedies_count)
comedies_count[:10].plot(kind='bar', xlabel="Genre", ylabel="# movies", title="Comedies by genre", grid='True')
plt.show()


comedies_scifi = movies[is_sf & is_comedy]
com_sf_counts = comedies_scifi['genres'].value_counts()
print(com_sf_counts)
com_sf_counts[:10].plot(kind='bar', xlabel="Genre", ylabel="# movies", title="Comedies & SF by genre", grid='True')
plt.show()

# 3. Merging dataframes for more in-depth analysis