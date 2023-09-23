#!/usr/bin/env python
# coding: utf-8

# # **DIABETES DISEASE ANALYSIS**

# In[29]:


import os
from IPython.display import Image
Image('Desktop/image.png')


# ### Diabetes can look and feel different for women. Identifying and treating the unique symptoms and risks of women with diabetes may lead to a better quality of life.
# 
# Diabetes is a group of metabolic diseases in which a person has high levels of blood glucose — also known as blood sugar — due to problems making or using the hormone insulin. Your body needs insulin to make and use energy from the carbohydrates you consume.
# 
# There are three common types:
# 
# Type 1 diabetes: Your body can’t make insulin due to autoimmune dysfunction. 
# Type 2 diabetes: This is the most common and occurs when your body is unable to properly use insulin. 
# Gestational diabetes: This is caused by pregnancy.

# #### In this project, we are going to analyse Pima Indians Diabetes dataset to identify root cause features of diabetes disease occurring into women body.

# In[30]:


Image('Desktop/images.jpeg')


# # Import Libraries

# In[31]:


# Importing Liabraries

import warnings
import pandas as pd
import numpy as np
import seaborn as sns
from matplotlib import pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from sklearn.neighbors import KNeighborsClassifier  
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score


# In[32]:


# Import Dataset

Diabetes_data = pd.read_csv("C:/Users/admin/Downloads/MeriSkill Internship/Project 2 MeriSKILL/diabetes.csv")


# In[33]:


# Read Dataset

Diabetes_data


# In[34]:


# Check columns from Dataset

Diabetes_data.info()


# In[35]:


# Check null values

Diabetes_data.isnull().sum()


# In[36]:


# Check duplicate data

duplicate_rows = Diabetes_data[Diabetes_data.duplicated()]
print("Duplicate Records", duplicate_rows.shape)


# # Feature Selection
# 
# Feature selection/extraction is an important step in many machine-learning tasks, including classification, regression, and clustering. It involves identifying and selecting the most relevant features (also known as predictors or input variables) from a dataset while discarding the irrelevant or redundant ones.
# 
# Here, Glucose and BMI are most relevant features.

# In[49]:


Features = ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin',
       'BMI', 'DiabetesPedigreeFunction', 'Age']


# In[50]:


from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import mutual_info_classif
from matplotlib import pyplot
 
 # split into input (X) and output (y) variables
X = Diabetes_data.iloc[:, :-1]
Y = Diabetes_data.iloc[:, -1]
 

# feature selection
def select_features(X_train, y_train, X_test):
 # configure to select all features
 fs = SelectKBest(score_func=mutual_info_classif, k='all')
 # learn relationship from training data
 fs.fit(X_train, y_train)
 # transform train input data
 X_train_fs = fs.transform(X_train)
 # transform test input data
 X_test_fs = fs.transform(X_test)
 return X_train_fs, X_test_fs, fs

X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.33, random_state=1)
# feature selection
X_train_fs, X_test_fs, fs = select_features(X_train, y_train, X_test)
# what are scores for the features
for i in range(len(fs.scores_)):
 print('Feature %d: %f' % (i, fs.scores_[i]))
# plot the scores
pyplot.pie([i for i in range(len(fs.scores_))], fs.scores_, labels=Features) 
y = np.array(Features)
pyplot.show()


# In[39]:


fs.index = X_train.columns
fs.index


# # UNIVARIATE ANALYSIS
# Univariate analysis explores each variable in a data set, separately. It looks at the range of values, as well as the central tendency of the values. It describes the pattern of response to the variable. It describes each variable on its own. Descriptive statistics describe and summarize data.

# In[40]:


p = Diabetes_data.hist(figsize = (17,20))


# # BIVARIATE ANALYSIS
# ### Analyse output variable with other/independent variable.
# 
# 
# Bivariate analysis is stated to be an analysis of any concurrent relation between two variables or attributes. This study explores the relationship of two variables as well as the depth of this relationship to figure out if there are any discrepancies between two variables and any causes of this difference.

# **1.High Blood Pressure:** Blood pressure target is usually below 140/90mmHg for people with diabetes or below 150/90mmHg if you are aged 80 years or above. For some people with kidney disease the target may be below 130/80mmHg.
# 
# **2.DiabetesPedigreeFunction:** PCA4: the most significant variables are “Diabetes Pedigree Function” with a value equal to 0.879, and “Diabetes” with a value equal to -0.227 and “Glucose” with an amount equal to - 0.292.
# 
# **3.Insulin:** Using commercial assays, normal fasting insulin levels range between 5 and 15 µU/mL but with more sensitive assays normal fasting insulin should be lower than 12 µU/mL. Obese subjects have increased values, while very high circulating levels are found in patients with severe insulin resistance.
# 
# **4.Age:** Age is a key factor in type 2 diabetes risk. Most people with type 2 diabetes receive a diagnosis at ages 45–64. Race, ethnicity, and socioeconomic factors can also affect a person's risk of developing the condition.
# 
# **5.Blood Pressure:** A normal blood pressure level is less than 120/80 mmHg. No matter your age, you can take steps each day to keep your blood pressure in a healthy range.
# 
# **6.Skin Thickness:** Skin thickness (epidermal surface to dermal fat inter- face), which is primarily determined by collagen con- tent, is greater in insulin-dependent diabetes mellitus (IDDM) patients who have been diabetic for >10 yr (11,12). This possibly reflects increased collagen cross- linkage and reduced collagen turnover (2,3)
# 
# **7.BMI:** Body Mass Index is a simple calculation using a person's height and weight. The formula is BMI = kg/m2 where kg is a person's weight in kilograms and m2 is their height in metres squared. A BMI of 25.0 or more is overweight, while the healthy range is 18.5 to 24.9. BMI applies to most adults 18-65 years.
# 
# **8.Pregnancies:** Pregnancy can make some diabetes health problems worse. To help prevent this, your health care team may recommend adjusting your treatment before you get pregnant.

# In[41]:


f = plt.figure(figsize=(10,22))
ax = f.add_subplot(122)
ax2 = f.add_subplot(122)
plt.subplot(2,2,1)
sns.swarmplot(x=Diabetes_data['Outcome'], y=Diabetes_data['Age'])
plt.subplot(2,2,2)
sns.swarmplot(x=Diabetes_data['Outcome'],y=Diabetes_data['DiabetesPedigreeFunction'])
plt.subplot(2,2,3)
sns.swarmplot(x=Diabetes_data['Outcome'], y=Diabetes_data['Pregnancies'])
plt.subplot(2,2,4)
sns.swarmplot(x=Diabetes_data['Outcome'], y=Diabetes_data['Glucose'])


# In[42]:


f = plt.figure(figsize=(10,13))
ax = f.add_subplot(122)
ax2 = f.add_subplot(122)
plt.subplot(2,2,1)
sns.swarmplot(x=Diabetes_data['Outcome'], y=Diabetes_data['Insulin'])
plt.subplot(2,2,2)
sns.swarmplot(x=Diabetes_data['Outcome'], y=Diabetes_data['BMI'])
plt.subplot(2,2,3)
sns.swarmplot(x=Diabetes_data['Outcome'], y=Diabetes_data['BloodPressure'])
plt.subplot(2,2,4)
sns.swarmplot(x=Diabetes_data['Outcome'], y=Diabetes_data['SkinThickness'])


# In[43]:


Diabetes_data.columns
X_Col = ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin',
       'BMI', 'DiabetesPedigreeFunction', 'Age']


# # Implementation of K-Nearest Neighbour
# 
# KNN: K Nearest Neighbor is one of the fundamental algorithms in machine learning. Machine learning models use a set of input values to predict output values. KNN is one of the simplest forms of machine learning algorithms mostly used for classification. It classifies the data point on how its neighbor is classified.

# #### We usually use Euclidean distance to calculate the nearest neighbor. If we have two points (x, y) and (a, b). The formula for Euclidean distance (d) will be
# 
# d = d=√((x2-x1)²+(y2-y1)²) 

# ###  Performing KNN by splitting to train and test set:

# In[44]:


x= Diabetes_data.iloc[:,0:13].values 
y= Diabetes_data['Outcome'].values
from sklearn.model_selection import train_test_split  
x_train, x_test, y_train, y_test= train_test_split(x, y, test_size= 0.25, random_state=0)
from sklearn.preprocessing import StandardScaler    
st_x= StandardScaler()    
x_train= st_x.fit_transform(x_train)    
x_test= st_x.transform(x_test)  


# ### Checking for the best value of  k:

# In[45]:


error = []
# Calculating error for K values between 1 and 30
for i in range(1, 30):
    knn = KNeighborsClassifier(n_neighbors=i)
    knn.fit(x_train, y_train)
    pred_i = knn.predict(x_test)
    error.append(np.mean(pred_i != y_test))
plt.figure(figsize=(12, 6))
plt.plot(range(1, 30), error, color='red', linestyle='dashed', marker='o',
         markerfacecolor='blue', markersize=10)
plt.title('Error Rate K Value')
plt.xlabel('K Value')
plt.ylabel('Mean Error')
print("Minimum error:-",min(error),"at K =",error.index(min(error))+1)


# ### Apply K-NN Algorithm:

# In[46]:


classifier= KNeighborsClassifier(n_neighbors=7)  
classifier.fit(x_train, y_train)
y_pred= classifier.predict(x_test) 
from sklearn.metrics import confusion_matrix  
cm= confusion_matrix(y_test, y_pred)
cm


# In[47]:


print(accuracy_score(y_test,y_pred))
print(precision_score(y_test, y_pred))
print(recall_score(y_test, y_pred))


# ## Confussion Metrix Output

# In[48]:


# output valuess in Confussion metrix.

group_names = ['True Neg','False Pos','False Neg','True Pos']
group_counts = ["{0:0.0f}".format(value) for value in
                cm.flatten()]
group_percentages = ["{0:.2%}".format(value) for value in
                     cm.flatten()/np.sum(cm)]
labels = [f"{v1}\n{v2}\n{v3}" for v1, v2, v3 in
          zip(group_names,group_counts,group_percentages)]
labels = np.asarray(labels).reshape(2,2)
sns.heatmap(cm, annot=labels, fmt='', cmap='Blues')


# Here, we got 98% accuracy of prediction.
# 190 persons having diabetes out of 192.

# In[ ]:




