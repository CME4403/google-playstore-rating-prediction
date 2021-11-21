import numpy as np 
import pandas as pd
from matplotlib import pyplot as plt

# Read dataset
df = pd.read_csv('./dataset/googleplaystore_expanded.csv')

# print(df.describe())

# Detect missing values
# print(df.isna().sum())

# Drop unnecessary columns
df.drop('App Id',inplace=True,axis=1)
df.drop('App Name',inplace=True,axis=1)
df.drop('Currency',inplace=True,axis=1)
df.drop('Developer Email',inplace=True,axis=1)
df.drop('Developer Id',inplace=True,axis=1)
df.drop('Developer Website',inplace=True,axis=1)
df.drop('Privacy Policy',inplace=True,axis=1)
df.drop('Scraped Time',inplace=True,axis=1)

## minimum and maximum install are dropped because we use just installs attribute
df.drop('Minimum Installs',inplace=True,axis=1)
df.drop('Maximum Installs',inplace=True,axis=1)

## released and minimum android is dropped because they've lots of missing values
df.drop('Minimum Android',inplace=True,axis=1)
df.drop('Released',inplace=True,axis=1)

#print(df.head())

## drop all rows having rating and rating count values are null
df = df.dropna(subset=['Rating', 'Rating Count'])
print(df.isna().sum())

# Detect Outliers


# Normalization

''' 
* last update, size, price, rating count, last updated and installs have continuous values
* category, rating, free(bool), content rating, ad supported(bool), in app purchases(bool) have discrete values 
'''

# Binning


# Sampling


# Visualization
