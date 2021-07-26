# -*- coding: utf-8 -*-

"""

@author: Maram Salameh
ISE 298


"""

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from pandas.plotting import scatter_matrix
from imblearn.over_sampling import SMOTE
from sklearn.cluster import AgglomerativeClustering
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import  StratifiedKFold
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.linear_model import LogisticRegression
from statistics import mean
from imblearn.under_sampling import CondensedNearestNeighbour
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.metrics import f1_score
from sklearn.model_selection import GridSearchCV




#### read data set
df= pd.read_csv('tri_2019_ca.csv')
df2 = pd.read_csv('tri_2018_ca.csv')


#data wrangling/ cleaning


df = df[['12. LATITUDE', '13. LONGITUDE',
          '19. INDUSTRY SECTOR CODE', '34. CHEMICAL', '38. CLEAN AIR ACT CHEMICAL',
          '40. METAL', '41. METAL CATEGORY', '42. CARCINOGEN', '44. UNIT OF MEASURE',
          '100. 6.2 - TOTAL TRANSFER','101. TOTAL RELEASES']]

df2 = df2[['12. LATITUDE', '13. LONGITUDE',
          '15. INDUSTRY SECTOR CODE', '30. CHEMICAL', '33. CLEAN AIR ACT CHEMICAL',
          '35. METAL', '36. METAL CATEGORY', '37. CARCINOGEN', '39. UNIT OF MEASURE',
          '95. 6.2 - TOTAL TRANSFER','96. TOTAL RELEASES']]

 
##### rename columns
colnames = ['LATITUDE', 'LONGITUDE',
          'INDUSTRY SECTOR CODE', 'CHEMICAL', 'CLEAN AIR ACT CHEMICAL',
          'METAL', 'METAL CATEGORY', 'CARCINOGEN', 'UNIT OF MEASURE', 'TOTAL TRANSFER',
          'TOTAL RELEASES']

df.columns= colnames
df2.columns = colnames



### merge datframes 
df= df.append(df2)


### drop duplicates 
#df.drop_duplicates(inplace= True) 

#### change col values from yes and no to 1 and 0
df['METAL'].replace(('YES', 'NO'), (1, 0), inplace=True)
df['CLEAN AIR ACT CHEMICAL'].replace(('YES', 'NO'), (1, 0), inplace=True)
#df['CARCINOGEN'].replace(('YES', 'NO'), (1, 0), inplace=True)
df['UNIT OF MEASURE'].replace(('Pounds', 'Grams'), (1, 0), inplace=True)

##drop instances where releases = 0
#df.drop(df[df['TOTAL RELEASES'] == 0].index, inplace = True)

### keep chemicals types with occurances more than 50
df = df[df['CHEMICAL'].map(df['CHEMICAL'].value_counts()) >= 50]
df.dropna(inplace=True)

#exploratory analysis 
df.shape
df.dtypes
df.head()
df.tail()
df['CHEMICAL'].describe()
df['CHEMICAL'].value_counts()


### change data type of chemical to category
data_types_dict = {'CHEMICAL': 'category'}
df = df.astype(data_types_dict)
df.dtypes

#ammonia = df[df['CHEMICAL']=='AMMONIA']
#ammonia.hist()
#scatter_matrix(ammonia)


### plot long/lat
#plt.figure(figsize = (15,8))
#sns.scatterplot(df['LATITUDE'], df['LONGITUDE'])


##convert to radian
df['LATITUDE'] = np.radians(df['LATITUDE'])
df['LONGITUDE'] = np.radians(df['LONGITUDE'])

## cluster
# creates 5 clusters fron long/lat using hierarchical clustering.
agc = AgglomerativeClustering(n_clusters =5, affinity='euclidean', linkage='ward')
df['LOCATION CLUSTER'] = agc.fit_predict(df[['LATITUDE','LONGITUDE']])

# Assigning numerical values and store in another column
df['CHEMICAL CATEGORY'] = df['CHEMICAL'].cat.codes
df


#drop transformed columns 
df.drop(['LATITUDE', 'LONGITUDE', 'CHEMICAL'],1, inplace=True)


#Train and evaluate models – try different models and parameters and cross validation

###features
X = np.array(df.drop(['CARCINOGEN'], 1))
#X = preprocessing.scale(X)
X
len(X)

###target
y = np.array(df['CARCINOGEN'])
y
len(y)



# apply over sampling or undersampling 
##define oveesampling method
oversample = SMOTE()
#transform the dataset
X, y = oversample.fit_resample(X, y)

###undersample
# define the undersampling method
undersample = CondensedNearestNeighbour(n_neighbors=2)
# transform the dataset
X, y = undersample.fit_resample(X, y)

#X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)
#len(X_train)
#len(X_test)
#len(y_train)
#len(y_test)

################################
        
##### train and evaluate models with cross valiadtion


##### cross validation
cv = StratifiedKFold(n_splits=50)


# Model 1: Random Forest Classifier
classifier = RandomForestClassifier(n_estimators=30, random_state=0)


# Evaluate model
scores_rf = cross_val_score(classifier, X, y, scoring='accuracy', cv=cv, n_jobs=-1)
rf_scoresf1 = cross_val_score(classifier, X, y, scoring='f1_weighted', cv=cv, n_jobs=-1)
#rf_stdf1 = cross_val_score(classifier, X, y, scoring='f1_weighted', cv=cv, n_jobs=-1).std()
rf_recall_w = cross_val_score(classifier, X, y, scoring='recall_weighted', cv=cv, n_jobs=-1)
rf_weighted_precision = cross_val_score(classifier, X, y, scoring='precision_weighted', cv=cv, n_jobs=-1)

# Report performance
print(mean(scores_rf)) #0.9925990338164251
print(mean(rf_scoresf1)) #0.9925282418864311
print(mean(rf_recall_w)) #0.9925990338164251
print(mean(rf_weighted_precision)) #0.9928940029089398


##Model 2
lgr = LogisticRegression()

# Evaluate Model
lgr_scores = cross_val_score(lgr, X, y, scoring='accuracy', cv=cv, n_jobs=-1)
lgr_scoresf1 = cross_val_score(lgr, X, y, scoring='f1_weighted', cv=cv, n_jobs=-1)
lgr_recall_w = cross_val_score(lgr, X, y, scoring='recall_weighted', cv=cv, n_jobs=-1)
lgr_weighted_precision = cross_val_score(lgr, X, y, scoring='precision_weighted', cv=cv, n_jobs=-1)

# Report performance
print(mean(lgr_scores)) #0.6574975845410628
print(mean(lgr_scoresf1)) #0.5352132288719689
print(mean(lgr_recall_w)) #0.5352132288719689
print(mean(lgr_weighted_precision)) #0.5503453986367384


##model 3
lda = LinearDiscriminantAnalysis()

# Evaluate Model
lda_scores = cross_val_score(lda, X, y, scoring='accuracy', cv=cv, n_jobs=-1)
lda_scoresf1 = cross_val_score(lda, X, y, scoring='f1_weighted', cv=cv, n_jobs=-1)
lda_recall_w = cross_val_score(lda, X, y, scoring='recall_weighted', cv=cv, n_jobs=-1)
lda_weighted_precision = cross_val_score(lda, X, y, scoring='precision_weighted', cv=cv, n_jobs=-1)

# Report performance
print(mean(lda_scores)) 
print(mean(lda_scoresf1)) 
print(mean(lda_recall_w))
print(mean(lda_weighted_precision)) 


#### Model 4: K Nearest Neighbor 
knn=KNeighborsClassifier(algorithm='auto', leaf_size=30, metric='minkowski',
 metric_params=None, n_jobs=1, n_neighbors=2, p=2,
 weights='uniform')

# Evaluate Model
knn_scores = cross_val_score(knn, X, y, scoring='accuracy', cv=cv, n_jobs=-1)
knn_scoresf1 = cross_val_score(knn, X, y, scoring='f1_weighted', cv=cv, n_jobs=-1)
#knn_stdf1 = cross_val_score(knn, X, y, scoring='f1_weighted', cv=cv, n_jobs=-1).std()
knn_recall_w = cross_val_score(knn, X, y, scoring='recall_weighted', cv=cv, n_jobs=-1)
knn_weighted_precision = cross_val_score(knn, X, y, scoring='precision_weighted', cv=cv, n_jobs=-1)

# Report performance
print(mean(knn_scores)) 
print(mean(knn_scoresf1)) 
print(mean(knn_recall_w)) 
print(mean(knn_weighted_precision))
