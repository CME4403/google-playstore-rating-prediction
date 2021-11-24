import numpy as np 
import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns
from sklearn import preprocessing
from sklearn.preprocessing import Normalizer, StandardScaler

plt.style.use('seaborn')

# Helper function: Draw Hist Plot
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

# Helper function: Draw Box Plot
def draw_boxplot(df, feature, vers):
  sns.boxplot(x=feature, data=df)
  title = feature + ' (' + vers + ' dropping outliers)'
  plt.title(title);
  plt.show()

# Missing Values
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
    
# Detect Outliers
def detect_and_drop_outliers(feature, df):
  q1 = df[feature].quantile(0.25)
  q3 = df[feature].quantile(0.75)  
  iqr = q3 - q1
  lower_bound = q1 - (iqr * 1.5)
  upper_bound = q3 + (iqr * 1.5)
  return df[~( (df[feature] < lower_bound) | (df[feature] > upper_bound) )]

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


  # Handle Missing Values
  missing_values(df)
  df_clean = df.copy()

  ### Continuous Features ###

  # Size Column 
  ''' size values that corresponds to 'Varies with device' with 'NaN' '''
  df_clean['Size'] = df_clean['Size'].replace('Varies with device', 'NaN', regex=True)
  df_clean['Size'] = df_clean['Size'].replace('nan', 'NaN', regex=True)

  ''' decimal point ',' to '.' '''
  df_clean['Size'] = df_clean['Size'].str.replace(',','.')

  ''' convert unit '''
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
  
  ''' fix units of Size '''
  df_clean['Size'] = size
  df_clean['Size'] = df_clean['Size'].astype(float)

  ''' Handling Size Missing Values '''
  df_clean['Size']=df_clean['Size'].fillna(df_clean["Size"].mode()[0])

  # Installs Column
  df_clean.Installs = df_clean.Installs.str.replace(',','',regex=True)
  df_clean.Installs = df_clean.Installs.str.replace('+','',regex=True)
  df_clean.Installs = df_clean.Installs.str.replace('Free','0',regex=True)
  df_clean['Installs'] = pd.to_numeric(df_clean['Installs'])
  
  #  Normalization
  ''' Continuous Features are: 'Size', 'Installs', 'Minimum Installs', 'Maximum Installs', 'Rating Count'  '''
  cont_features = ['Size', 'Installs', 'Minimum Installs', 'Maximum Installs', 'Rating Count']

  '''
  for feature in cont_features:
   
    minmax_scale = preprocessing.MinMaxScaler(feature_range=(0,1))
    will_normalize_feature = np.array(df_clean[feature])
    scaled_feature = minmax_scale.fit_transform(will_normalize_feature)

    scaled_feature = np.array(scaled_feature)
    df_clean[feature] = StandardScaler().fit_transform(scaled_feature)
  '''

  normalizer = Normalizer(norm="l2")
  df_clean[cont_features] = normalizer.transform(df_clean[cont_features])


  # Find and Clean Outliers
  '''  boxplot before drop outliers '''
  for feature in cont_features:
    draw_boxplot(df_clean, feature, 'before')

  ''' drop outliers '''
  for feature in cont_features:
    df_clean = detect_and_drop_outliers(feature, df_clean)

  '''  boxplot after drop outliers '''
  for feature in cont_features:
    draw_boxplot(df_clean, feature, 'after')


  ### Categorical Features ###

  # Content Rating
  '''  print(df_clean['Content Rating'].value_counts()) '''
  df_clean['Content Rating'] = df_clean['Content Rating'].replace('Unrated',"Everyone")

  '''  Cleaning other values just to include Everyone, Teens and Adult '''
  df_clean['Content Rating'] = df_clean['Content Rating'].replace('Mature 17+',"Adults")
  df_clean['Content Rating'] = df_clean['Content Rating'].replace('Adults only 18+',"Adults")
  df_clean['Content Rating'] = df_clean['Content Rating'].replace('Everyone 10+',"Everyone")
  
  # Release Date and Update Date  
  ''' Burada COVID19 1 aralıkta başlamış kabul edilmiş ama ben onu 2020 de başlamış olarak kabul ediyorum. '''
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
  
  ''' append covid feature '''
  df_clean['Covid'] = covid
  print(df_clean['Covid'])
  
  ''' drop released Date '''
  df_clean.drop('Released', inplace=True,axis=1)

  print("Dataset information", df_clean.info())
  
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

  ### Visualization ###

  # Rating Column
  '''
  num_plots(df_clean,'Rating','App rating distribution','Rating') '''

  # Categories
  '''
  sns.countplot(x='Category', data=df_clean, order = df_clean['Category'].value_counts().index)
  plt.xticks(rotation=90);
  plt.xlabel('')
  plt.title('App category counts');
  plt.show() '''

  # App Types
  '''
  sns.countplot(x='Free', data=df_clean)
  plt.title('Free or not')
  plt.xlabel('App type')
  plt.show()
  '''
 
  # Last Updated
  '''
  plt.figure(figsize=(10,4))
  sns.histplot(x='Last Updated', data=df_clean)
  plt.show()
  '''

  # Content Rating
  '''
  sns.countplot(x='Content Rating', data=df_clean)
  plt.title('Content Rating')
  plt.xticks(rotation=60)
  plt.show()
  '''

  # Minimum Android
  '''
  sns.countplot(x='Minimum Android', data=df_clean)
  plt.title('Minimum Android')
  plt.xticks(rotation=60)
  plt.show()
  '''

  # Released
  '''
  sns.countplot(x='Released', data=df_clean)
  plt.title('Released')
  plt.xticks(rotation=60)
  plt.show()
  '''

  # Ad Supported
  '''
  sns.countplot(x='Ad Supported', data=df_clean)
  plt.title('Ad Supported')
  plt.xticks(rotation=60)
  plt.show()
  '''

  # In App Purchases
  '''
  sns.countplot(x='In App Purchases', data=df_clean)
  plt.title('In App Purchases')
  plt.xticks(rotation=60)
  plt.show()
  '''

  # Editors Choice
  '''
  sns.countplot(x='Editors Choice', data=df_clean)
  plt.title('Editors Choice')
  plt.xticks(rotation=60)
  plt.show()
  '''

  # Correlation Matrix
  sns.heatmap(df_clean.corr(), annot=True, cmap='Blues')
  plt.title('Correlation Matrix')
  plt.show()
  

if __name__ == "__main__":
    main()