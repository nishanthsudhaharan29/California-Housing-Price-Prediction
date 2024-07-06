#!/usr/bin/env python
# coding: utf-8

# # EDA and Feature Engineering

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


# In[2]:


housing = pd.read_csv(r"C:\Users\nisha\Downloads\archive (7)\housing.csv")
housing


# In[3]:


housing.info()


# The column total bedrooms has around 200 missing values. Dropping the missing values as it has sufficient data even after dropping.

# In[4]:


# dropping the missing values and checking it again
housing.dropna(inplace = True)
housing.isna().sum()


# In[5]:


# creating dummy values
dummies =pd.get_dummies(housing['ocean_proximity'])
housing = housing.join(dummies)
housing['ocean_proximity'].value_counts()


# In[6]:


# drop the categorical variable and one of the dummies
housing.drop(columns = ['ocean_proximity','NEAR BAY'],axis=1, inplace = True)


# In[7]:


# check the distribution of the variables
housing.hist(figsize = (15,8))


# From the histograms, it can be seen that the variables total_rooms, total_bedrooms, population, households are skewed.

# In[8]:


# check the correlation between the variables
plt.figure(figsize = (15,8))
sns.heatmap(housing.corr(), annot = True,cmap = 'coolwarm')


# In[9]:


# convert those skewed variables into logs of those variables
housing['total_rooms'] = np.log(housing['total_rooms']) +1 
housing['total_bedrooms'] = np.log(housing['total_bedrooms']) +1 
housing['population'] = np.log(housing['population']) +1 
housing['households'] = np.log(housing['households']) +1 


# In[10]:


# check the distribution again after converting into log
housing.hist(figsize = (15,8))


# As the number of rooms and the number of bedrooms depend on the size of the neighborhood block, we create new variables with the ratio of the number of rooms and the number of households, the number of bedrooms and number of household in the block.

# In[11]:


# feature engineering
housing['bedroom_ratio'] = housing['total_bedrooms'] / housing['total_rooms']
housing['bedroom_avg'] = housing['total_bedrooms'] / housing['households']
housing['room_avg'] = housing['total_rooms'] / housing['households']


# In[12]:


sns.scatterplot(x=housing['latitude'], y=housing['longitude'], hue = housing['median_house_value'],palette = 'coolwarm')
plt.show()


# The above scatter plot shows that the neighborhoods close to the ocean cost more.

# From the correlation matrix, we can see that the total rooms and the total bedrooms are very highly correlated with each other. Population is also highly correlated with the discussed. So we remove total_rooms and total_bedrooms from the model and decide to use the room ratio and the bedroom ratio instead.

# In[13]:


housing.drop(columns = ['total_rooms','total_bedrooms'],axis =1, inplace = True)


# The processed data is as follows:

# In[14]:


print(housing)


# In[15]:


# splitting the dependent and independent variables 
y = housing['median_house_value']
x = housing.drop(columns = ['median_house_value'],axis =1)


# In[16]:


# split the data into training set and testing set
from sklearn.model_selection import train_test_split
xtrain, xtest, ytrain, ytest = train_test_split(x,y,test_size = 0.2, random_state = 109)


# In[17]:


# normalize the variables 
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
scaler.fit(xtrain)
xtrain = scaler.transform(xtrain)
xtest = scaler.transform(xtest)


# # Model Building and Evaluation

# Linear Regression

# In[18]:


from sklearn.linear_model import LinearRegression
reg = LinearRegression()
reg.fit(xtrain, ytrain)
predictions = reg.predict(xtest)
print("The score for the model is:", reg.score(xtest,ytest))
print(reg.intercept_)
print(reg.coef_)


# The above linear regression model explains only 64.27% of the total variation in the mean houcing price of the block. Let's try other approaches.

# Random Forest Regressor

# In[19]:


from sklearn.ensemble import RandomForestRegressor
forest = RandomForestRegressor()
forest.fit(xtrain, ytrain)
print(forest.score(xtest, ytest))


# The above random forest regression model explains about 76.59% of the total variation in the mean houcing price of the block.

# The score is improved, but let us find the optimal parameters.

# In[20]:


from sklearn.model_selection import GridSearchCV
forest = RandomForestRegressor()
params = {'n_estimators': [100,200,300], 'min_samples_split':[2,4,6]}
search = GridSearchCV(forest, params, cv = 5, scoring = 'neg_mean_squared_error', return_train_score = True)
search.fit(xtrain, ytrain)


# In[21]:


print(search.best_estimator_)
print(search.best_estimator_.score(xtest, ytest))


# The above random forest regression model explains about 81.63% of the total variation in the mean houcing price of the block.

# In[ ]:




