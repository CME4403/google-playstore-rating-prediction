import numpy as np 
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
from sklearn import metrics
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import KFold,GridSearchCV, learning_curve, train_test_split, cross_val_score
from sklearn.naive_bayes import BernoulliNB, CategoricalNB, GaussianNB, MultinomialNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import Normalizer, StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

plt.style.use('seaborn')

##İleride eğitim seti için ayırmakta kullanabliriz
#train_data,test_data=train_test_split(playstore,test_size=0.15,random_state=42)

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

def knn_method(data):
  Install=[]
  #Target Feature minimize
  for i in data['Installs']:
    if i <=500:
      Install.append('Low')
    elif i>500 and i<=100000:
        Install.append('Medium')
    elif i>100000:
        Install.append('High')
    else:
        Install.append('Unranked')
        
  data.drop('Installs',inplace=True,axis=1)    
  data['Installs'] = Install 

  features = data.drop('Installs', axis=1)
  target = data.Installs

  standardizer = StandardScaler()
  features = standardizer.fit(features).transform(features)
  features_train, features_test, target_train, target_test = train_test_split(features, target,test_size=0.2 ,random_state=1)

  # Create a KNN classifier
  knn_model = KNeighborsClassifier(n_neighbors=5, n_jobs=-1).fit(features_train, target_train)
  # Create a pipeline
  pipe = Pipeline([("standardizer", standardizer), ("knn", knn_model)])
  # Create space of candidate values
  search_space = [{"knn__n_neighbors": np.arange(1,10)}]
  # Create grid search
  classifier = GridSearchCV(pipe,search_space,cv=5).fit(features_train, target_train)
  # Best neighborhood size (k)
  best_k = classifier.best_estimator_.get_params()["knn__n_neighbors"]
  print(best_k)
  # KNN model and prediction with best K
  knn = KNeighborsClassifier(n_neighbors=best_k)
  knn.fit(features_train, target_train)
  target_pred = knn.predict(features_test)
  print(metrics.accuracy_score(target_test, target_pred))
  print('Class prediction is: %s', target_pred)
  # View probability of observation 
  prob = knn.predict_proba(features_test)
  print('Probability of prediction: %s', prob)
  # Evaulating algorithm
  print("Accuracy of model: ",metrics.accuracy_score(target_test, target_pred))
  print(confusion_matrix(target_test, target_pred))
  print(classification_report(target_test, target_pred))
  print(knn.score(features_test, target_test))

  cv = KFold(n_splits=5, shuffle=True, random_state=1)
  cv_results = cross_val_score(knn,features_test, target_test, scoring='accuracy', cv=cv, n_jobs=-1, error_score='raise')
  print('Accuracy of KNN classifier with k-fold:', cv_results)

  ''' Error Rate for 1-10 K values'''
  error_rate = []
  for i in range(1,10):
    knn = KNeighborsClassifier(n_neighbors=i)
    knn.fit(features_train, target_train)
    pred_i = knn.predict(features_test)
    error_rate.append(np.mean(pred_i != target_test))

  plt.figure(figsize=(10,6))
  plt.plot(range(1,10),error_rate,color='blue', linestyle='dashed', 
          marker='o',markerfacecolor='red', markersize=10)
  plt.title('Error Rate vs. K Value')
  plt.xlabel('K')
  plt.ylabel('Error Rate')
  plt.show()
  print("Minimum error:-",min(error_rate),"at K =",error_rate.index(min(error_rate)))

  ''' Accuracy Rate for 1-10 K Values'''
  acc = []
  for i in range(1,10):
      neigh = KNeighborsClassifier(n_neighbors = i).fit(features_train, target_train)
      yhat = neigh.predict(features_test)
      acc.append(metrics.accuracy_score(target_test, yhat))
      
  plt.figure(figsize=(10,6))
  plt.plot(range(1,10),acc,color = 'blue',linestyle='dashed', 
          marker='o',markerfacecolor='red', markersize=10)
  plt.title('accuracy vs. K Value')
  plt.xlabel('K')
  plt.ylabel('Accuracy')
  plt.show()
  print("Maximum accuracy:-",max(acc),"at K =",acc.index(max(acc)))
  return False

def logistic_reg(data):
  features = data.drop('Installs', axis=1)
  target = data.Installs
  # Create training and test set
  features_train, features_test, target_train, target_test = train_test_split(features, target, random_state=1)
  # Create logistic regression
  classifier = LogisticRegression()
  # Train model and make predictions
  target_predicted = classifier.fit(features_train, target_train).predict(features_test)
  # Create confusion matrix
  matrix = confusion_matrix(target_test, target_predicted)
  # Create a classification report
  print(classification_report(target_test, target_predicted))
  return False

def linear_reg(data):
  features = data.drop('Installs', axis=1)
  target = data.Installs
  standardizer = StandardScaler()
  features = standardizer.fit(features).transform(features)
  features_train, features_test, target_train, target_test = train_test_split(features, target,test_size=0.2 ,random_state=1)
  # Create logistic regression object
  linear_reg = LinearRegression()
  # Train model
  model = linear_reg.fit(features_train, target_train)
  # View the intercept (w_zero)
  print(model.intercept_)
  # View the feature coefficients (weights)
  print(model.coef_)

  predictions = model.predict(features_test)
  plt.scatter(target_test, predictions)
  plt.show()
  print('MAE:', metrics.mean_absolute_error(target_test, predictions))
  return False

# Processing
def processing_methods():
  df = pd.read_csv('./clean_googleplaystore_dataset.csv')
  df.drop('Unnamed: 0',inplace=True,axis=1)
  df = df.sample(n=10000,replace="False")
  onehotencodeddata = pd.get_dummies(df, columns = ['Category','Content Rating','AppRating','Type','month'])
  method_id = input("Enter method type: ")
  return {
    'knn': knn_method,
    'linear': linear_reg,
    'logistic': logistic_reg
  }[method_id](onehotencodeddata)


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
  #print("Dataset information", df.info())  

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
  ''' Continuous Features are: 'Size', 'Minimum Installs', 'Maximum Installs',  '''
  cont_features = ['Size', 'Minimum Installs', 'Maximum Installs']

  '''
  for feature in cont_features:
   
    minmax_scale = preprocessing.MinMaxScaler(feature_range=(0,1))
    will_normalize_feature = np.array(df_clean[feature])
    scaled_feature = minmax_scale.fit_transform(will_normalize_feature)

    scaled_feature = np.array(scaled_feature)
    df_clean[feature] = StandardScaler().fit_transform(scaled_feature)
  '''

  ''' least squares normalizer form '''
  normalizer = Normalizer(norm="l2")
  df_clean[cont_features] = normalizer.transform(df_clean[cont_features])


  # Find and Clean Outliers
  '''  boxplot before drop outliers 
  for feature in cont_features:
    draw_boxplot(df_clean, feature, 'before')
  '''
  ''' drop outliers '''
  for feature in cont_features:
    df_clean = detect_and_drop_outliers(feature, df_clean)
  
  '''  boxplot after drop outliers 
  for feature in cont_features:
    draw_boxplot(df_clean, feature, 'after')
  '''


  ### Categorical Features ###

  # Content Rating
  '''  print(df_clean['Content Rating'].value_counts()) '''
  df_clean['Content Rating'] = df_clean['Content Rating'].replace('Unrated',"Everyone")

  '''  Cleaning other values just to include Everyone, Teens and Adult '''
  df_clean['Content Rating'] = df_clean['Content Rating'].replace('Mature 17+',"Adults")
  df_clean['Content Rating'] = df_clean['Content Rating'].replace('Adults only 18+',"Adults")
  df_clean['Content Rating'] = df_clean['Content Rating'].replace('Everyone 10+',"Everyone")
  
  conditions = [
    (df_clean['Rating'] == 0) & (df_clean['Rating Count'] == 0),
    (df_clean['Rating'] >= 0) & (df_clean['Rating'] <= 1) & (df_clean['Rating Count'] != 0),
    (df_clean['Rating'] > 1) & (df_clean['Rating'] <= 2) & (df_clean['Rating Count'] != 0),
    (df_clean['Rating'] > 2) & (df_clean['Rating'] <= 3) & (df_clean['Rating Count'] != 0),
    (df_clean['Rating'] > 3) & (df_clean['Rating'] <= 4) & (df_clean['Rating Count'] != 0),
    (df_clean['Rating'] > 4) & (df_clean['Rating'] <= 5) & (df_clean['Rating Count'] != 0)
    ]

  # create a list of the values we want to assign for each condition
  values = ['Unranked', 'Very Bad', 'Bad', 'Not Bad','Good','Very Good']

  # create a new column and use np.select to assign values to it using our lists as arguments
  df_clean['AppRating'] = np.select(conditions, values)
  
  df_clean = df_clean.drop(['Rating','Rating Count'], axis = 1)
  
  # print(len(df_clean['Category'].value_counts()))
  # Release Date and Update Date  
  ''' Burada COVID19 1 aralıkta başlamış kabul edilmiş ama ben onu 2020 de başlamış olarak kabul ediyorum. '''
  df_clean['Released'] = pd.to_datetime(df_clean['Released'], format='%b %d, %Y',infer_datetime_format=True, errors='coerce') 
  df_clean['Last Updated'] = pd.to_datetime(df_clean['Last Updated'], format='%b %d, %Y',infer_datetime_format=True, errors='coerce') 
  u = df_clean.select_dtypes(include=['datetime'])
  md=df_clean['Released'].median()
  df_clean[u.columns]=u.fillna(md)

  covid= []
  
  for i in pd.DatetimeIndex(df_clean['Released']).year:
      if i >= 2020:
        covid.append(True)
      else:
        covid.append(False)

  lastupdate = []
  
  for i in pd.DatetimeIndex(df_clean['Last Updated']).year:
      if i > 2020:
        lastupdate.append(True)
      else:
        lastupdate.append(False)
 
  #append Up to Date
  df_clean['Up to Date'] = lastupdate
  
  #append covid feature
  df_clean['Covid'] = covid
  
  # Minimum Android Version
  min_andr_ver = []
  # print(df_clean.info())
  ver_mode = df_clean['Minimum Android'].mode()[0]
  for i in df_clean['Minimum Android']:
    ver = i.split()
    if ver =='Varies with Device' or ver =='Var':
      min_andr_ver.append(ver_mode)
    else:
      min_andr_ver.append(ver[0][:3])
          
  df_clean['Minimum Android'] = min_andr_ver
  df_clean['Minimum Android'] = df_clean['Minimum Android'].str.replace('Var','4.1',regex=True)
  df_clean['Size'] = df_clean['Size'].astype(float)

  #Free
  df_clean['Type'] = np.where(df_clean['Free'] == True,'Free','Paid')
  df_clean.drop(['Free'],inplace=True,axis=1)
  
  
  #print("Dataset information",df_clean.info())  
  # print(df_clean['Minimum Android'].value_counts())
  # open the file in the write mode
  df_clean.drop('Last Updated',inplace=True,axis=1)
  
  months= []
  
  for i in pd.DatetimeIndex(df_clean['Released']).month:
      if i == 1:
        months.append('January')
      elif i == 2:
        months.append('February')
      elif i == 3:
        months.append('March')
      elif i == 4:
        months.append('April')
      elif i == 5:
        months.append('May')
      elif i == 6:
        months.append('June')  
      elif i == 7:
        months.append('July')
      elif i == 8:
        months.append('August')
      elif i == 9:
        months.append('September')
      elif i == 10:
        months.append('October')
      elif i == 11:
        months.append('November')
      elif i == 12:
        months.append('December')        

  #append Up to Date
  df_clean['month'] = months
  
  df_clean.drop('Released',inplace=True,axis=1)
  df_clean.drop('Minimum Installs',inplace=True,axis=1)
  df_clean.drop('Maximum Installs',inplace=True,axis=1)

  df_clean.to_csv('./clean_googleplaystore_dataset.csv')

  #
  #
  # METHODS AND EVAULATIONS
  #
  #
  # processing_methods() # This method includes KNN, Linear and Logistic Regression Models
  
  ''' CLASSIFICATION '''
  df = pd.read_csv('./clean_googleplaystore_dataset.csv')

  df.drop('Unnamed: 0',inplace=True,axis=1)
  df.drop('Size',inplace=True,axis=1)

  #df = df.sample(n=1000000 ,replace="False")
  df = df.sample(n=100 ,replace="False")
  df['Installs']  = df['Installs'].astype(str)
  onehotencodeddata = pd.get_dummies(df, columns = ['Category','Content Rating','AppRating','Type','month','Minimum Android'])

  """Split"""

  features = onehotencodeddata.drop('Installs',axis = 1)
  target = onehotencodeddata['Installs']
  #class_names = target.unique()

  features_train, features_test, target_train, target_test = train_test_split(features, target,test_size=0.2 ,random_state=1)


  """Decision Tree"""

  """
  decisiontreetainscore = []
  decisiontreetestscore = []
  maxdepth = []

  for i in range(20,40):
      maxdepth.append( i)   
      clf = DecisionTreeClassifier(random_state = 0,max_depth=i).fit(features_train, target_train)
      decisiontreetainscore.append(clf.score(features_train, target_train))
      decisiontreetestscore.append(clf.score(features_test, target_test))
      print(i)
  """

  param_grid = {'max_depth': [10, 20, 30, 40, 50,60,70,80,90],
                'criterion':['gini','entropy']}

  grid = GridSearchCV(DecisionTreeClassifier(random_state=1), param_grid=param_grid, refit = True, verbose = 3,scoring="accuracy")
  grid.fit(features_train, target_train)

  # print best parameter after tuning
  print(grid.best_params_)
  # print how our model looks after hyper-parameter tuning
  print(grid.best_estimator_)
  grid_predictions = grid.predict(features_test)

  """
  clf = DecisionTreeClassifier(random_state = 0,max_depth=20).fit(features_train, target_train)
  plt.plot(grid.param_grid['maxdepth'], grid.cv_results_, color='r', label='Train Score')
  plt.plot(grid.param_grid['maxdepth'], grid.cv_results_, color='g', label='Test Score')

  plt.xlabel("Depth Size")
  plt.ylabel("Train Performance")
  plt.title("Train Performance by Depth Size")
  """
  """
  clf = DecisionTreeClassifier(random_state = 0,max_depth=35).fit(features_train, target_train)
  #print('Accuracy of Decision Tree classifier on training set(max depth): {:.2f}'.format(clf.score(features_train, target_train)))
  print('Accuracy of Decision Tree classifier on test set(max depth): {:.2f}' .format(clf.score(features_test, target_test)))

  clf1 = DecisionTreeClassifier(random_state = 0).fit(features_train, target_train)
  #print('Accuracy of Decision Tree classifier on training set(not max depth): {:.2f}'.format(clf1.score(features_train, target_train)))
  print('Accuracy of Decision Tree classifier on test set(not max depth): {:.2f}' .format(clf1.score(features_test, target_test)))

  cv = KFold(n_splits=5, shuffle=True, random_state=1)

  cv_results_max_depth = cross_val_score(clf,features_test, target_test, scoring='accuracy', cv=cv, n_jobs=-1, error_score='raise')
  print("Accuracy of Decision Tree classifier with n-fold on training set(max depth):", format(100 * cv_results_max_depth.mean(), ".2f") + "%")

  cv_results = cross_val_score(clf1,features_test, target_test, scoring='accuracy', cv=cv, n_jobs=-1, error_score='raise')
  print("Accuracy of Decision Tree classifier with n-fold on training set(not max depth):", format(100 * cv_results.mean(), ".2f") + "%")

  target_predicted = clf.predict(features_test)
  print('Predicted Class: %s' % target_predicted[0])

  # Create confusion matrix
  matrix = confusion_matrix(target_test, target_predicted)
  # Create pandas dataframe
  dataframe = pd.DataFrame(matrix, index=class_names, columns=class_names)


  # Create heatmap
  sns.heatmap(dataframe,annot=True, cbar=None, cmap="Blues")
  plt.title("Confusion Matrix"), plt.tight_layout()
  plt.ylabel("True Class"), plt.xlabel("Predicted Class")
  plt.show()

  # Create a classification report
  print(classification_report(target_test,target_predicted,target_names=class_names))

  train_sizes, train_scores, test_scores = learning_curve( DecisionTreeClassifier(), features, target, cv=5, scoring='accuracy', n_jobs=-1, train_sizes=np.linspace( 0.01, 1.0, 50))
  # Create means and standard deviations of training set scores
  train_mean = np.mean(train_scores, axis=1)
  train_std = np.std(train_scores, axis=1)
  # Create means and standard deviations of test set scores
  test_mean = np.mean(test_scores, axis=1)
  test_std = np.std(test_scores, axis=1)
  # Draw lines
  plt.plot(train_sizes, train_mean, '--', color="#111111", label="Training score")
  plt.plot(train_sizes, test_mean, color="#111111", label="Cross-validation score")
  # Draw bands
  plt.fill_between(train_sizes, train_mean - train_std,
  train_mean + train_std, color="#DDDDDD")
  plt.fill_between(train_sizes, test_mean - test_std,
  test_mean + test_std, color="#DDDDDD")
  # Create plot
  plt.title("Learning Curve")
  plt.xlabel("Training Set Size"), plt.ylabel("Accuracy Score"),
  plt.legend(loc="best")
  plt.tight_layout()
  plt.show()
  """

  """"Random Forest
  # Create random forest classifier object
  randomforest=RandomForestClassifier(random_state=0, n_jobs=-1,max_depth=39)
  # Train model
  model = randomforest.fit(features_train, target_train)

  randomforest1=RandomForestClassifier(random_state=0, n_jobs=-1)
  # Train model
  model2 = randomforest1.fit(features_train, target_train)

  # evaluate the model
  cv = KFold(n_splits=5, shuffle=True, random_state=1)

  cv_results = cross_val_score(model2,features_test, target_test, scoring='accuracy', cv=cv, n_jobs=-1, error_score='raise')

  cv_results_max_depth = cross_val_score(model2,features_test, target_test, scoring='accuracy', cv=cv, n_jobs=-1, error_score='raise')

  # report performance
  print("Accuracy of Random Forest Tree classifier with n-fold on training set(not set max depth):", format(100 * cv_results.mean(), ".2f") + "%")

  print("Accuracy of Random Forest Tree classifier with n-fold on training set(max depth):", format(100 * cv_results_max_depth.mean(), ".2f") + "%")

  # report performance
  print('Accuracy of Random Forest Tree classifier with split on test set(max depth set): {:.2f}' .format(model.score(features_test, target_test)))

  # report performance

  print('Accuracy of Random Forest Tree classifier with split on test set(not set max depth ): {:.2f}' .format(model2.score(features_test, target_test)))


  target_predicted = model.predict(features_test)
  print('Predicted Class: %s' % target_predicted[0])

  # Create confusion matrix
  matrix = confusion_matrix(target_test, target_predicted)
  # Create pandas dataframe
  dataframe = pd.DataFrame(matrix, index=class_names, columns=class_names)

  # Create a classification report
  print(classification_report(target_test,target_predicted,target_names=class_names))


  """
  #
  #
  ''' NAIVE BAYES '''
  #
  #
  df = pd.read_csv('./clean_googleplaystore_dataset.csv')
  df.drop('Unnamed: 0',inplace=True,axis=1)
  #df = df.sample(n=500000,replace="False")
  df = df.sample(n=100 ,replace="False")
  df['Installs']  = df['Installs'].astype(str)

  """ Minimize the Install target attribute values
  Install=[]
  #Target Feature minimize
  for i in df['Installs']:
    if i <=500:
      Install.append('Low')
    elif i>500 and i<=100000:
        Install.append('Medium')
    elif i>100000:
        Install.append('High')
    else:
        Install.append('Unranked')
        
  df.drop('Installs',inplace=True,axis=1)    
  df['Installs']=Install
  """

  #print(df['Installs'].value_counts())
  onehotencodeddata = pd.get_dummies(df, columns = ['Category','Content Rating','AppRating','Type','month'])
  features = onehotencodeddata.drop('Installs',axis = 1)
  target = onehotencodeddata.Installs
  class_names = target.unique()

  """Split"""
  features_train, features_test, target_train, target_test = train_test_split(features, target,test_size=0.2 ,random_state=1)

  #Bernoulli
  bnb = BernoulliNB()
  bnb.fit(features_train, target_train)
  print('Accuracy of Bernolli classifier on training set: {:.2f}'.format(bnb.score(features_train, target_train)))
  print('Accuracy of Bernolli classifier on test set: {:.2f}'.format(bnb.score(features_test, target_test)))

  #######N-fold Cross Validation
  cv = KFold(n_splits=5, shuffle=True, random_state=1)
  cv_results = cross_val_score(bnb,features_test, target_test, scoring='accuracy', cv=cv, n_jobs=-1, error_score='raise')

  print("Accuracy of Bernoulli Naive Bayes classifier with n-fold on test set:", format(100 * cv_results.mean(), ".2f") + "%")

  """
  #Multinominal
  bnb = MultinomialNB()
  bnb.fit(features_train, target_train)
  print('Accuracy of GNB classifier on training set: {:.2f}'.format(bnb.score(features_train, target_train)))
  print('Accuracy of GNB classifier on test set: {:.2f}'.format(bnb.score(features_test, target_test)))
  """
  """
  #Gaussian
  bnb = GaussianNB()
  bnb.fit(features_train, target_train)
  print('Accuracy of GNB classifier on training set: {:.2f}'.format(bnb.score(features_train, target_train)))
  print('Accuracy of GNB classifier on test set: {:.2f}'.format(bnb.score(features_test, target_test)))

  """
  """
  #CategoricalNB
  bnb = CategoricalNB()
  bnb.fit(features_train, target_train)
  print('Accuracy of GNB classifier on training set: {:.2f}'.format(bnb.score(features_train, target_train)))
  print('Accuracy of GNB classifier on test set: {:.2f}'.format(bnb.score(features_test, target_test)))
  """
  #
  #
  ''' SVM '''
  #
  #
  df = pd.read_csv('./clean_googleplaystore_dataset.csv')
  df.drop('Unnamed: 0',inplace=True,axis=1)
  df = df.sample(n=1000,replace="False")
  df['Installs']  = df['Installs'].astype(str)

  onehotencodeddata = pd.get_dummies(df, columns = ['Category','Content Rating','AppRating','Type','month'])
  features = onehotencodeddata.drop('Installs',axis = 1)
  target = onehotencodeddata.Installs

  """Split"""
  features_train, features_test, target_train, target_test = train_test_split(features, target,test_size=0.2 ,random_state=1)


  # defining parameter range auto= 1/n_features
  param_grid = {'C': [0.1, 1, 10, 100, 1000],
                'gamma': [0.001, 0.01, 0.1, 1, 10, 100,'auto'],
                'kernel': ['linear']}

  grid = GridSearchCV(SVC(), param_grid, refit = True, verbose = 3,scoring="accuracy")

  grid.fit(features_train, target_train)

  # print best parameter after tuning
  print(grid.best_params_)
  
  # print how our model looks after hyper-parameter tuning
  print(grid.best_estimator_)


  grid_predictions = grid.predict(features_test)

  print(grid_predictions)
  """
  features = onehotencodeddata.drop('Installs',axis = 1)
  target = onehotencodeddata.Installs

  features_train, features_test, target_train, target_test = train_test_split(features, target,test_size=0.2 ,random_state=1)
  class_names = target_test.unique()

  svm = SVC()
  svm.fit(features_train, target_train)
  print('Accuracy of SVM classifier on training set: {:.2f}'.format(svm.score(features_train, target_train)))
  print('Accuracy of SVM classifier on test set: {:.2f}'.format(svm.score(features_test, target_test)))

  target_predicted = svm.predict(features_test)
  # print classification report
  print(classification_report(target_test, target_predicted))"""

  # 
  #
  ### Visualization ###
  #
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


  # Released
  ''' Visualization
  sns.countplot(x='Released', data=df_clean)
  plt.title('Released')
  plt.xticks(rotation=60)
  plt.show()
 
  released_date_install=pd.concat([df_clean['Installs'],df_clean['Released']],axis=1)
  plt.figure(figsize=(15,12))
  released_date_plot=released_date_install.set_index('Released').resample('3M').mean()
  released_date_plot.plot()
  plt.title('Released date Vs Installs',fontdict={'size':20,'weight':'bold'})
  plt.plot()
  '''

  #Covid-19 
  """
  plt.pie(df_clean['Covid'].value_counts(),radius=3,autopct='%0.2f%%',explode=[0.2,0.5],colors=['#ffa500','#0000a0'],labels=['Before Covid','Covid'],startangle=90,textprops={'fontsize': 30})
  plt.title('Covid-19 Impact on Applications',fontdict={'size':20,'weight':'bold'})
  plt.plot()
  """
  #Up to Date
  """
  plt.pie(df_clean['Up to Date'].value_counts(),radius=3,autopct='%0.2f%%',explode=[0.2,0.5],colors=['#ffa500','#0000a0'],labels=['No','Yes'],startangle=90,textprops={'fontsize': 30})
  plt.title('Is the app up to date?',fontdict={'size':20,'weight':'bold'})
  plt.plot()
  """

    #Type vs Install
  '''
  plt.figure(figsize=(18,18))
  ax = sns.countplot(df_clean['Installs'],hue=df_clean['Type']);
  plt.title("Number of Installs in different Types ")

  plt.xticks(fontsize=10,fontweight='bold',rotation=45,ha='right');
  plt.show()
  '''

  # Last Updated
  '''
  plt.figure(figsize=(10,4))
  sns.histplot(x='Last Updated', data=df_clean)
  plt.show()
 
  lastupdate_install=pd.concat([df_clean['Installs'],df_clean['Last Updated']],axis=1)
  plt.figure(figsize=(15,12))
  released_date_plot=lastupdate_install.set_index('Last Updated').resample('3M').mean()
  released_date_plot.plot()
  plt.title('Last Update Vs Installs',fontdict={'size':20,'weight':'bold'})
  plt.plot()
  '''

  # Content Rating
  '''
  age_install = df_clean.groupby('Content Rating')['Minimum Installs'].mean()

  plt.axes().set_facecolor("white")
  plt.rcParams.update({'font.size': 12, 'figure.figsize': (5, 4)})
  plt.ylabel('Category')
  plt.xlabel('Installs per 10 million')
  age_install.sort_index().plot(kind="barh", title='Average Number of Installs per Content Rating');
  plt.gca().invert_yaxis()
  plt.savefig("Age rating", transparent=False, bbox_inches="tight")
  '''

  #Categories vs Install
  #draw a boxplot map to observe app's ratings among different categories
  """
  category_rating = df.groupby(['Category'])['Installs'].count()

  plt.figure(figsize=(15,10))
  sns.barplot(category_rating.index, category_rating.values)
  plt.title('Number of Installs Per Category')
  plt.xlabel('Category')
  plt.ylabel('Installs')
  plt.xticks(fontsize=10,fontweight='bold',rotation=45,ha='right');
  """
"""
  f, ax = plt.subplots(2,2,figsize=(10,15))

  ax[0,0].hist(df_clean.Rating, range=(3,5))
  ax[0,0].set_title('Ratings Histogram')
  ax[0,0].set_xlabel('Ratings')

  d = df_clean.groupby('Category')['Rating'].mean().reset_index()
  ax[0,1].scatter(d.Category, d.Rating)
  ax[0,1].set_xticklabels(d.Category.unique(),rotation=90)
  ax[0,1].set_title('Mean Rating per Category')

  ax[1,1].hist(df_clean.Size, range=(0,100),bins=10, label='Size')
  ax[1,1].set_title('Size Histogram')
  ax[1,1].set_xlabel('Size')

  d = df_clean.groupby('Size')['Installs'].mean().reset_index()
  ax[1,0].scatter(d.Size, d.Installs)
  ax[1,0].set_xticklabels(d.Size.unique(),rotation=90)
  ax[1,0].set_title('Mean Install per Size')
  f.tight_layout()
"""
"""
  # Correlation Matrix
  sns.heatmap(df_clean.corr(), annot=True, cmap='Blues')
  plt.title('Correlation Matrix')
  plt.show()
  """

if __name__ == "__main__":
    main()
