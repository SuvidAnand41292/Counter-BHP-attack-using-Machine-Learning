
# coding: utf-8

# In[1]:


#IMPORTING LIBRARIES


# In[2]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().magic('matplotlib inline')


# In[3]:


# GET THE DATA


# In[4]:


DataFrame = pd.read_csv('OBS-Network-Data.csv')


# In[5]:


#Quick Look at the Data Structure


# In[6]:


DataFrame.describe()


# In[7]:


DataFrame.head(5)


# In[8]:


DataFrame.info()


# In[9]:


# handling missing values
sns.heatmap(DataFrame.isnull(),yticklabels=False,cbar=False,cmap='viridis')


# In[10]:


import matplotlib.pyplot as plt
DataFrame.hist(bins=50, figsize=(20,15))
plt.show()


# In[11]:


#TRAIN TEST SPLIT


# In[12]:


from sklearn.model_selection import train_test_split


# In[13]:


DataFrame.columns


# In[14]:


X = DataFrame.drop('class', axis = 1)
y = DataFrame['class']


# In[15]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)


# In[16]:


print(len(X_train), "train +", len(X_test), "test")


# In[17]:


# DATA VISUALISATION


# In[18]:


Y_train = pd.DataFrame(y_train.replace(to_replace=["'NB-No Block'", 'NB-Wait', "'No Block'", 'Block'], value=[0,1,2,3]))


# In[19]:


# corelation


# In[20]:


corr_matrix = X_train.corr()


# In[21]:


corr_matrix["percentage of lost byte rate"].sort_values(ascending=False)


# In[22]:


sns.heatmap(X_train.corr(), linewidths=1)


# In[23]:


sns.pairplot(X_train.drop(labels = ['Packet size_byte'], axis = 1))


# In[24]:


sns.countplot(x='class',data=Y_train)


# In[25]:


# DATA CLEANING


# In[26]:


df_xtr = X_train.drop(labels = ['node ','percentage of lost packet rate', 'percentage of lost byte rate','packet received rate','Packet size_byte'], axis = 1)


# In[27]:


# missing values
df_xtr['packet lost'] = pd.to_numeric(df_xtr['packet lost'], errors = 'coerce')

sns.heatmap(df_xtr.isnull(),yticklabels=False,cbar=False,cmap='viridis')


# In[28]:


# replacing missing values with mean value
df_xtr['packet lost'].fillna(value=df_xtr['packet lost'].mean(),inplace = True)


# In[29]:


# changing categorical value with quantitative value
from sklearn.preprocessing import LabelEncoder
encoder = LabelEncoder()
df_xtr['node status'] = encoder.fit_transform(df_xtr['node status'])


# In[30]:


#feature scaling
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
scaler.fit(df_xtr)


# In[31]:


# box plot
df_bp = df_xtr.copy()
df_bp['class'] = Y_train
for i in ['utilised BW Rate ', 'packet drop rate', 'full bandwidth',
       'avg delay time/sec', 'used BW', 'lost BW', 'packet transmitted',
       'packet received', 'packet lost', 'transmitted byte', 'received byte',
       '10 run avg drop rate', '10 run avg BW', '10 run delay', 'node status',
       'flood status',]:
    plt.figure(figsize=(6,6))

    sns.boxplot(x='class' , y= i , data = df_bp, palette='winter')


# In[32]:


#handling outliers
df_xtr.drop(['avg delay time/sec', '10 run delay','flood status' ], axis = 1, inplace = True)


# In[33]:


#TRAINING MODEL


# In[34]:


# logistic regression
from sklearn.linear_model import LogisticRegression
logmodel = LogisticRegression()
clf_1 = logmodel.fit(df_xtr,y_train)


# In[35]:


from sklearn.metrics import classification_report,confusion_matrix


# In[36]:


#K-Fold CROSS VALIDATION logestic regression

from sklearn.model_selection import cross_val_score
scores_1 = cross_val_score(clf_1, df_xtr, y_train, cv=10)
print("Accuracy: %0.2f (+/- %0.2f)" % (scores_1.mean(), scores_1.std() * 2))                                              


# In[37]:


# random forest 
from sklearn.ensemble import RandomForestClassifier
rfc = RandomForestClassifier(n_estimators=100)
clf_2 = rfc.fit(df_xtr, y_train)


# In[38]:


#K-Fold CROSS VALIDATION random forest

scores_2 = cross_val_score(clf_2, df_xtr, y_train, cv=10)
print("Accuracy: %0.2f (+/- %0.2f)" % (scores_2.mean(), scores_2.std() * 2))                                              


# In[39]:


#LDA & QDA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
qda = QuadraticDiscriminantAnalysis()
lda = LinearDiscriminantAnalysis() 
clf_3 = lda.fit(df_xtr, y_train)
clf_4 = qda.fit(df_xtr, y_train)


# In[40]:


#K-Fold CROSS VALIDATION lda

scores_3 = cross_val_score(clf_3, df_xtr, y_train, cv=10)
print("Accuracy: %0.2f (+/- %0.2f)" % (scores_3.mean(), scores_3.std() * 2))


# In[41]:


#K-Fold CROSS VALIDATION qda

scores_4 = cross_val_score(clf_4, df_xtr, y_train, cv=10)
print("Accuracy: %0.2f (+/- %0.2f)" % (scores_4.mean(), scores_4.std() * 2))


# In[42]:


#TEST SET PERFORMANCE


# In[43]:


df_xtst = X_test.drop(labels = ['node ','percentage of lost packet rate', 'percentage of lost byte rate',
                                'packet received rate','Packet size_byte','avg delay time/sec', 
                                '10 run delay','flood status'], axis = 1)

df_xtst['packet lost'] = pd.to_numeric(df_xtst['packet lost'], errors = 'coerce')
df_xtst['packet lost'].fillna(value=df_xtst['packet lost'].mean(),inplace = True)

encoder = LabelEncoder()
df_xtst['node status'] = encoder.fit_transform(df_xtst['node status'])


# In[44]:


#logistic
predictions_1 = logmodel.predict(df_xtst)
print(confusion_matrix(y_test,predictions_1))
print(classification_report(y_test,predictions_1))


# In[45]:


#random forest
predictions_2 = rfc.predict(df_xtst)
print(confusion_matrix(y_test,predictions_2))
print(classification_report(y_test,predictions_2))


# In[46]:


#lda
predictions_3 = lda.predict(df_xtst)
print(confusion_matrix(y_test,predictions_3))
print(classification_report(y_test,predictions_3))


# In[47]:


#qda
predictions_4 = qda.predict(df_xtst)
print(confusion_matrix(y_test,predictions_4))
print(classification_report(y_test,predictions_4))


# In[48]:


#fine tuning and optimization


# In[49]:


from sklearn.decomposition import PCA
from sklearn.preprocessing import scale


# In[50]:


X = df_xtr.values
X = scale(X)
pca = PCA(n_components=13)
pca.fit(X)

#The amount of variance that each PC explains
var= pca.explained_variance_ratio_

#Cumulative Variance explains
var1=np.cumsum(np.round(pca.explained_variance_ratio_, decimals=4)*100)

print (var1)


# In[51]:


plt.plot(var1)


# In[52]:


#Looking at above plot I'm taking 6 variables
pca = PCA(n_components=6)
pca.fit(X)
X1=pca.fit_transform(X)

print(X1)


# In[53]:


clf_5 = logmodel.fit(X1,y_train)
scores_5 = cross_val_score(clf_5, X1, y_train, cv=10)
print("Accuracy: %0.2f (+/- %0.2f)" % (scores_5.mean(), scores_9.std() * 2))


# In[54]:


clf_6 = rfc.fit(X1,y_train)
scores_6 = cross_val_score(clf_6, X1, y_train, cv=10)
print("Accuracy: %0.2f (+/- %0.2f)" % (scores_6.mean(), scores_6.std() * 2))


# In[55]:


clf_7 = qda.fit(X1,y_train)
scores_7 = cross_val_score(clf_7, X1, y_train, cv=10)
print("Accuracy: %0.2f (+/- %0.2f)" % (scores_7.mean(), scores_7.std() * 2))


# In[56]:


clf_8 = lda.fit(X1,y_train)
scores_8 = cross_val_score(clf_8, X1, y_train, cv=10)
print("Accuracy: %0.2f (+/- %0.2f)" % (scores_8.mean(), scores_8.std() * 2))


# In[57]:


X_t=X_test.drop(['node status', 'packet lost'], axis = 1).values
X_t = scale(X_t)
pca = PCA(n_components=13)
pca.fit(X)

#The amount of variance that each PC explains
vari= pca.explained_variance_ratio_

#Cumulative Variance explains
var2=np.cumsum(np.round(pca.explained_variance_ratio_, decimals=4)*100)

print (var2)


# In[58]:


plt.plot(var2)


# In[59]:


pca = PCA(n_components=6)
pca.fit(X_t)
X2=pca.fit_transform(X_t)

print(X2)


# In[60]:


predictions_8 = logmodel.predict(X2)
print(confusion_matrix(y_test,predictions_8))
print(classification_report(y_test,predictions_8))


# In[61]:


predictions_9 = qda.predict(X2)
print(confusion_matrix(y_test,predictions_9))
print(classification_report(y_test,predictions_9))


# In[62]:


predictions_10 = lda.predict(X2)
print(confusion_matrix(y_test,predictions_10))
print(classification_report(y_test,predictions_10))


# In[63]:


predictions_10 = rfc.predict(X2)
print(confusion_matrix(y_test,predictions_10))
print(classification_report(y_test,predictions_10))


# In[64]:


from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2


# In[65]:


names = ['utilised BW Rate ', 'packet drop rate', 'full bandwidth', 'used BW',
       'lost BW', 'packet transmitted', 'packet received', 'packet lost',
       'transmitted byte', 'received byte', '10 run avg drop rate',
       '10 run avg BW', 'node status']


# In[66]:


# feature selection using UNIVARIATE Selection
# feature extraction
test = SelectKBest(score_func=chi2, k=13)
fit = test.fit(df_xtr, y_train)
# summarize scores
np.set_printoptions(precision=3)
print(fit.scores_)
features = fit.transform(X)


# In[67]:


print(features[0:5,:])


# In[68]:


df_xtr.columns


# In[69]:


df_chi = df_xtr.copy()
df_chit = df_xtst.copy()


# In[70]:


df_chi.drop(['utilised BW Rate ', 'packet drop rate', 'full bandwidth', 'used BW',
       'lost BW', '10 run avg drop rate',
       '10 run avg BW', 'node status'], axis = 1, inplace = True)
df_chit.drop(['utilised BW Rate ', 'packet drop rate', 'full bandwidth', 'used BW',
       'lost BW', '10 run avg drop rate',
       '10 run avg BW', 'node status'], axis = 1, inplace = True)


# In[71]:


clf1 = logmodel.fit(df_chi, y_train)
pred1 = logmodel.predict(df_chit)

print(confusion_matrix(y_test,pred1))
print(classification_report(y_test,pred1))




# In[72]:


clf2 = rfc.fit(df_chi, y_train)
pred2 = rfc.predict(df_chit)

print(confusion_matrix(y_test,pred2))
print(classification_report(y_test,pred2))


# In[73]:


clf3 = qda.fit(df_chi, y_train)
pred3 = qda.predict(df_chit)

print(confusion_matrix(y_test,pred3))
print(classification_report(y_test,pred3))


# In[74]:


clf4 = lda.fit(df_chi, y_train)
pred4 = lda.predict(df_chit)

print(confusion_matrix(y_test,pred4))
print(classification_report(y_test,pred4))

