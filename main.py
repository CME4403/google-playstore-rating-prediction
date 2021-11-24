import numpy as np 
import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns

plt.style.use('seaborn')

# missing values
def missing_values(df):
    #check all missing values in dataset  
    print("Number of rows having null values in the dataset:")
    missing_info = (len(df[df.isnull().any(axis=1)]) / len(df) )*100
    print(len(df[df.isnull().any(axis=1)]),' which is ' ,round(missing_info,2) , '%')
    
    #check missing values in columns
    cols = df.columns[df.isnull().any()].to_list()
    print("Columns having null values are :",cols)

    for c in cols:
      missing_info=((df[c].isnull().sum()) / len(df[c]) )*100
      print(c,type(c),": ",df[c].isnull().sum(), "," ,round(missing_info,2) , '%')

    #drop if the missing value count is less than %1
    df.dropna(subset=['Installs'],inplace=True)
    df.dropna(subset=['Size'],inplace=True)
    df.dropna(subset=['Minimum Android'],inplace=True)
    df.dropna(subset=['Minimum Installs'],inplace=True)
    
    #Handling Rating and Rating Count missing values
    df['Rating']  = df['Rating'].astype(float)
    avg = round(df['Rating'].mean(),1)
    df['Rating'].fillna(avg,inplace=True)

    df['Rating Count']  = df['Rating Count'].astype(float)
    avg = round(df['Rating Count'].mean(),1)
    df['Rating Count'].fillna(avg,inplace=True)
    
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

  df.head()
  print("Dataset information", df.info())  

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
  # handle missing values
  missing_values(df)
  df_clean = df.copy()

  ### Numerical Features ###

  # Size Column
  # print(df_clean[~df_clean['Size'].str.contains('[k,M,G,Varies with device,nan]$', regex= True, na=False)]);
  
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
    elif i[-1] == 'k' or i[-1] == 'K':
      size.append(float(i[:-1])/1000)
    elif i[-1] == 'G' or i[-1] == 'g':
      size.append(float(i[:-1])*1000)
    else:
      size.append(float(i[:-1]))
  
  # fix units of Size
  df_clean['Size'] = size
  df_clean['Size'] = df_clean['Size'].astype(float)

  # Handling Size Missing Values
  df_clean['Size']=df_clean['Size'].fillna(df_clean["Size"].mode()[0])

  # Install
  df.Installs = df.Installs.str.replace(',','',regex=True)
  df.Installs = df.Installs.str.replace('+','',regex=True)
  df.Installs = df.Installs.str.replace('Free','0',regex=True)
  df['Installs'] = pd.to_numeric(df['Installs'])
  
  # Minimum Android Version
  min_andr_ver = []
  ver_mode = df_clean['Minimum Android'].mode()[0]
  for i in df_clean['Minimum Android']:
    ver = i.split()
    if ver =='Varies with Device':
      min_andr_ver.append(ver_mode)
    else:
      min_andr_ver.append(ver[0][:3])
          
  df_clean['Minimum Android'] = min_andr_ver
  df_clean['Size'] = df_clean['Size'].astype(float)
  
  # Content Rating

  # print(df_clean['Content Rating'].value_counts())
  df_clean['Content Rating'] = df_clean['Content Rating'].replace('Unrated',"Everyone")

  # Cleaning other values just to include Everyone, Teens and Adult 

  df_clean['Content Rating'] = df_clean['Content Rating'].replace('Mature 17+',"Adults")
  df_clean['Content Rating'] = df_clean['Content Rating'].replace('Adults only 18+',"Adults")
  df_clean['Content Rating'] = df_clean['Content Rating'].replace('Everyone 10+',"Everyone")
  
  # Release Date and Update Date  
 
  #Burada COVID19 1 aralıkta başlamış kabul edilmiş ama ben onu 2020 de başlamış olarak kabul ediyorum.
  df_clean['Released'] = df_clean['Released'].fillna("NaN")
  df_clean['Last Updated'] = df_clean['Last Updated'].fillna("NaN")
  release_date = []
  lastupdate_date = []
  
  for i in df_clean["Released"]:
      if i == 'NaN':
        release_date.append(None)
      else:
        x = i.split(", ")[1]
        release_date.append(int(x))
        
  for i in df_clean["Last Updated"]:
        x = i.split(", ")[1]
        lastupdate_date.append(int(x))   
          
  df_clean['Released'] = release_date
  df_clean['Released'] = round(df_clean['Released'].interpolate(method ='linear'))

  df_clean['Last Updated'] = lastupdate_date
  
  covid = []
  
  for i in df_clean["Released"]:
      if i >= 2020:
        covid.append(True)
      else:
        covid.append(False)

  df_clean.loc[(df_clean['Last Updated'] - df_clean['Released'] > 0),'Last Updated']=False
  df_clean.loc[(df_clean['Last Updated'] - df_clean['Released'] <= 0),'Last Updated']=True
  
  # append covid feature
  df_clean['Covid'] = covid
  print(df_clean['Covid'])
  
  # drop released Date
  df_clean.drop('Released', inplace=True,axis=1)

  print("Dataset information", df_clean.info())
  
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