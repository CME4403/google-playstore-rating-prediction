import numpy as np 
import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns
plt.style.use('seaborn')

# print(df.describe())
def num_plots(df, col, title, xlabel):
    fig, ax = plt.subplots(2, 1, sharex=True, figsize=(8,5),gridspec_kw={"height_ratios": (.2, .8)})
    ax[0].set_title(title,fontsize=18)
    sns.boxplot(x=col, data=df, ax=ax[0])
    ax[0].set(yticks=[])
    sns.histplot(x=col, data=df, ax=ax[1])
    ax[1].set_xlabel(xlabel, fontsize=16)
    plt.tight_layout()
    plt.show()

# Detect Outliers (for integers)
def indicies_of_outliers(x):
  q1, q3 = np.percentile(x, [25,75])
  iqr = q3 - q1
  lower_bound = q1 - (iqr * 1.5)
  upper_bound = q3 + (iqr * 1.5)
  return np.where((x > upper_bound) | (x < lower_bound))

# Normalization
def normalization():
  ''' 
  * last update, size, price, rating count, last updated and installs have continuous values
  * category, rating, free(bool), content rating, ad supported(bool), in app purchases(bool) have discrete values 
  '''
  # Binning
  # Sampling
  # Visualization
  print("hello")

def main():
  # read dataset
  df = pd.read_csv('./dataset/googleplaystore_expanded.csv')

  # drop unnecessary columns
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

  ## drop all rows having rating and rating count values are null
  df = df.dropna(subset=['Rating', 'Rating Count'])

  # drop duplicate values
  df.drop_duplicates(subset='App Name', inplace=True, ignore_index=True)
  df_clean = df


  ### Numerical Features ###

  # Size Column

  # print(df_clean[~df_clean['Size'].str.contains('[k,M,G,Varies with device]$', regex= True, na=False)])
  
  # size values that corresponds to Varies with device with 'NaN'
  df_clean['Size'] = df_clean['Size'].replace('Varies with device', 'NaN', regex=True)

  # decimal point ',' to '.'
  df_clean['Size'] = df_clean['Size'].str.replace(',','.')

  # convert unit
  size =[]

  for i in df_clean['Size']:
    if i == 'NaN':
      size.append('NaN')
    elif i[-1] == 'k':
      size.append(float(i[:-1])/1000)
    elif i[-1] == 'G':
      size.append(float(i[:-1])*1000)
    else:
      size.append(float(i[:-1]))

  # fix units of Size
  df_clean['Size'] = size
  df_clean['Size'] = df_clean['Size'].astype(float)
  
  num_plots(df_clean,'Size','App Size distribution','Size (MB)')
  
  print('Average app size is: ', df_clean['Size'].mean())
  print('Median app size is: ', df_clean['Size'].median())
  print('Mode app size is: ', df_clean['Size'].mode()[0])

  # Rating Column
  num_plots(df_clean,'Rating','App rating distribution','Rating')

  print('Average app rating is: ', df_clean['Rating'].mean())
  print('Median app rating is: ', df_clean['Rating'].median())
  print('Mode app rating is: ', df_clean['Rating'].mode()[0])
  #df_clean[df_clean['Rating'] <= 1.0]

  # Price Column
  print(df_clean['Price'].isnull().sum())

  # Installs Column
  #df_clean['Size'] = df_clean['Size'].str.replace(',','')
  #df_clean['Installs'] = df_clean['Installs'].str.replace('+','').astype(float)
  #sns.kdeplot(x='Installs', data=df_clean)


  ### Categorical Features ###
  # Categories
  sns.countplot(x='Category', data=df_clean, order = df_clean['Category'].value_counts().index)
  plt.xticks(rotation=90);
  plt.xlabel('')
  plt.title('App category counts');
  plt.show()

  # App Types
  sns.countplot(x='Free', data=df_clean)
  plt.title('Free or not')
  plt.xlabel('App type')
  plt.show()

  # Last Updated
  '''
  df_clean['Last Updated']=pd.to_datetime(df_clean['Last Updated'])
  plt.figure(figsize=(10,4))
  sns.histplot(x='Last Updated', data=df_clean)
  plt.show()

  df_clean['last_up_year']=df_clean['Last Updated'].dt.year
  plt.figure(figsize=(10,4))
  sns.histplot(x='last_up_year', data=df_clean)
  plt.show()
'''

  # detect missing values
  #print(df_clean.isna().sum())

  # find outliers
  '''
  * outliers are probably on these columns:  rating count[2], price[5], size[6], last updated[7]
  other features have values can be categorized.
  '''
  #features = df.to_numpy()

  #rating_outlier_indicies = indicies_of_outliers(features[:,2])
  #price_outlier_indicies = indicies_of_outliers(features[:,5])
  #size_outlier_indicies = indicies_of_outliers(features[:,6])
  #last_updated_outlier_indicies = indicies_of_outliers(features[:,7])

  '''
  print("rating_outlier_indicies \n")
  print(rating_outlier_indicies)
  print(len(rating_outlier_indicies[0]))
  print("\n price_outlier_indicies \n")
  print(price_outlier_indicies)
  print(len(price_outlier_indicies[0]))
  #print("\n size_outlier_indicies \n")
  #print(size_outlier_indicies)
  #print("\n last_updated_outlier_indicies \n")
  #print(last_updated_outlier_indicies)
  '''

if __name__ == "__main__":
    main()