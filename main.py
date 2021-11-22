import numpy as np 
import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns

plt.style.use('seaborn')

# print(df.describe())
def num_plots(df, col, title, xlabel):
    fig, ax = plt.subplots(2, 1, sharex=True, figsize=(8,5))
    ax[0].set_title(title,fontsize=18)
    
    mean_text = 'Mean of ', col, ' is: ', df[col].mean()
    median_text = 'Median of ', col, ' is: ', df[col].median()
    mode_text = 'Mode of ', col, ' is: ', df[col].mode()[0]
    '''
    plt.text(0, 1, mean_text, fontsize=12, bbox=dict(facecolor='red', alpha=0.5))
    plt.text(0, 3, median_text, fontsize=12, bbox=dict(facecolor='green', alpha=0.5))
    plt.text(0, 5, mode_text, fontsize=12, bbox=dict(facecolor='blue', alpha=0.5))
    '''
    print(mean_text, '\n', median_text, '\n', mode_text, '\n')

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

  # drop duplicate values
  df.drop_duplicates(subset='App Name', inplace=True, ignore_index=True)

  # drop unnecessary columns
  df.drop('App Id',inplace=True,axis=1)
  df.drop('App Name',inplace=True,axis=1)
  df.drop('Currency',inplace=True,axis=1)
  df.drop('Developer Email',inplace=True,axis=1)
  df.drop('Developer Id',inplace=True,axis=1)
  df.drop('Developer Website',inplace=True,axis=1)
  df.drop('Price',inplace=True,axis=1)
  df.drop('Privacy Policy',inplace=True,axis=1)
  df.drop('Scraped Time',inplace=True,axis=1)

  df_clean = df.copy()

  print(df_clean.columns)

  ## drop all rows having rating and rating count values are null (bunu silince size patliyor nedense bulamadim sebebini henuz)
  df_clean = df_clean.dropna(subset=['Rating', 'Rating Count', 'Installs'])

  #df_clean[df_clean['Rating'] <= 1.0]


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

  ### Numerical Features ###

  # Size Column
  #print(df_clean[~df_clean['Size'].str.contains('[k,M,G,Varies with device,nan]$', regex= True, na=False)]);
  
  # size values that corresponds to 'Varies with device' with 'NaN'
  df_clean['Size'] = df_clean['Size'].replace('Varies with device', 'NaN', regex=True)
  df_clean['Size'] = df_clean['Size'].replace('nan', 'NaN', regex=True)

  # decimal point ',' to '.'
  df_clean['Size'] = df_clean['Size'].str.replace(',','.')

  # convert unit
  size = []
  
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
  
  ''' Visualization
  # num_plots(df_clean,'Size','App Size distribution','Size (MB)')
  '''

  # Rating Count Column
  ''' Visualization 
  sns.boxplot(x='Rating Count', data=df_clean)
  plt.show()
  '''

  # Installs Column
  '''
  df_clean['Installs'] = df_clean['Installs'].str.replace('+','', regex=True)
  df_clean['Installs'] = df_clean['Installs'].str.replace(',','', regex=True).astype(float)
  df_clean.rename(columns={df_clean.columns[3]:'Installs(+)'}, inplace=True)
  #num_plots(df_clean,'Installs(+)','App Installing Count Distribution','Installs(+)')
  sns.boxplot(x='Installs(+)', data=df_clean)
  plt.show()

  ## Installs Info
  print('Total apps', len(df_clean))
  no_installs = [1e9, 1e8, 1e7, 1e6, 1e5, 1e4, 1e3, 1e2, 1e1]
  for n in no_installs:
    print('Number of apps with less than ' + str(n) + ' installs:', len(df_clean.loc[df_clean['Installs(+)']<n]))
  '''

  # Minimum Installs Column
  '''
  #num_plots(df_clean,'Minimum Installs','App Installing Count Distribution','Installs(+)')
  sns.boxplot(x='Minimum Installs', data=df_clean)
  plt.show()
  '''

  # Maximum Installs Column
  '''
  #num_plots(df_clean,'Maximum Installs','Maximum Installs Count Distribution','Installs(+)')
  sns.boxplot(x='Maximum Installs', data=df_clean)
  plt.show()
  '''

  ### Categorical Features ###

  # Rating Column
  ''' Visualization
  num_plots(df_clean,'Rating','App rating distribution','Rating') '''

  # Categories
  ''' Visualization
  sns.countplot(x='Category', data=df_clean, order = df_clean['Category'].value_counts().index)
  plt.xticks(rotation=90);
  plt.xlabel('')
  plt.title('App category counts');
  plt.show() '''

  # App Types
  ''' Visualization
  sns.countplot(x='Free', data=df_clean)
  plt.title('Free or not')
  plt.xlabel('App type')
  plt.show()
  '''

  # Last Updated
  # df_clean['Last Updated'] = pd.to_datetime(df_clean['Last Updated'])
  
  # add new last update year feature
  # df_clean['last_update_year'] = df_clean['Last Updated'].dt.year
  '''
  plt.figure(figsize=(10,4))
  sns.histplot(x='Last Updated', data=df_clean)
  plt.show()
  '''

  # Content Rating
  ''' Visualization
  sns.countplot(x='Content Rating', data=df_clean)
  plt.title('Content Rating')
  plt.xticks(rotation=60)
  plt.show()
  '''

  # Minimum Android
  ''' Visualization
  sns.countplot(x='Minimum Android', data=df_clean)
  plt.title('Minimum Android')
  plt.xticks(rotation=60)
  plt.show()
  '''

  # Released
  ''' Visualization
  sns.countplot(x='Released', data=df_clean)
  plt.title('Released')
  plt.xticks(rotation=60)
  plt.show()
  '''

  # Ad Supported
  ''' Visualization
  sns.countplot(x='Ad Supported', data=df_clean)
  plt.title('Ad Supported')
  plt.xticks(rotation=60)
  plt.show()
  '''

  # In App Purchases
  ''' Visualization
  sns.countplot(x='In App Purchases', data=df_clean)
  plt.title('In App Purchases')
  plt.xticks(rotation=60)
  plt.show()
  '''

  # Editors Choice
  ''' Visualization
  sns.countplot(x='Editors Choice', data=df_clean)
  plt.title('Editors Choice')
  plt.xticks(rotation=60)
  plt.show()
  '''

  # detect missing values
  #print(df_clean.isna().sum())

  # Correlation Matrix
  
  sns.heatmap(df_clean.corr(), annot=True, cmap='Blues')
  plt.title('Correlation Matrix')
  plt.show()
  

if __name__ == "__main__":
    main()