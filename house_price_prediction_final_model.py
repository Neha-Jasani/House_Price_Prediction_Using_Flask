#!/usr/bin/env python
# coding: utf-8

# # Project: House Price Prediction

# ## Dataset Features:
# - Price: The price of the house.
# - Area: The total area of the house in square feet.
# - Bedrooms: The number of bedrooms in the house.
# - Bathrooms: The number of bathrooms in the house.
# - Stories: The number of stories in the house.
# - Mainroad: Whether the house is connected to the main road (Yes/No).
# - Guestroom: Whether the house has a guest room (Yes/No).
# - Basement: Whether the house has a basement (Yes/No).
# - Hot water heating: Whether the house has a hot water heating system (Yes/No).
# - Airconditioning: Whether the house has an air conditioning system (Yes/No).
# - Parking: The number of parking spaces available within the house.
# - Prefarea: Whether the house is located in a preferred area (Yes/No).
# - Furnishing status: The furnishing status of the house (Fully Furnished, Semi-Furnished, Unfurnished).

# ## Step 1: Data Preprocessing and Exploration

# In[165]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score,mean_absolute_error,mean_squared_error
from sklearn.linear_model import LinearRegression
from sklearn.feature_selection import RFE

from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import GradientBoostingRegressor



# In[166]:


data  = pd.read_csv('Housing.csv')


# In[167]:


data.head()


# In[168]:


data.info()


# In[169]:


data.describe()


# In[170]:


data['price_per_sqft'] = data['price'] / data['area']


# In[171]:


data = data[(data['price_per_sqft'] >= 100) & (data['price_per_sqft'] <= 1000)]


# #### - Checking missing value/treatment of missing value:-

# In[172]:


data.isnull().sum()


# In[173]:


data.plot(kind='box', subplots=True, figsize=(22,20), layout=(3,3))
plt.show()


# In[174]:


min_max_values = data.agg({'price': ['min', 'max'],
                           'area': ['min', 'max'],
                           'bedrooms': ['min', 'max'],
                           'bathrooms': ['min', 'max'],
                           'parking': ['min', 'max'],
                           'price_per_sqft': ['min', 'max']})


# In[175]:


print(min_max_values)


# In[176]:


# Define lower and upper bounds for each numerical feature 
lower_bounds = {
    'price': 1750000,
    'area': 2400,  
    'bedrooms': 1,  
    'bathrooms': 1,
    'parking': 0,
    'price_per_sqft': 270.39555
}


# In[177]:


upper_bounds = {
    'price': 10150000,
    'area': 16200,  
    'bedrooms': 6,  
    'bathrooms': 3,
    'parking': 3,
    'price_per_sqft': 1000.00000
}


# In[178]:


# Apply capping to remove outliers
for feature, lower_bound in lower_bounds.items():
    data[feature] = np.where(data[feature] < lower_bound, lower_bound, data[feature])

for feature, upper_bound in upper_bounds.items():
    data[feature] = np.where(data[feature] > upper_bound, upper_bound, data[feature])


# #### - Visual Exploration

# In[179]:


# Visualize the distribution of the target variable 'Price'
sns.histplot(data['price'], bins=30)
plt.xlabel('Price')
plt.ylabel('Frequency')
plt.title('Distribution of House Prices')
plt.show()


# In[180]:


# Visualize correlations between numerical features
correlation_matrix = data.corr()
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm')
plt.title('Correlation Matrix')
plt.show()


# In[181]:


# Explore categorical features
plt.figure(figsize=(12, 6))
sns.boxplot(x='bedrooms', y='price', data=data)
plt.xlabel('Number of Bedrooms')
plt.ylabel('Price')
plt.title('Price vs. Number of Bedrooms')
plt.xticks(rotation=45)
plt.show()


# In[182]:


# Visualize the distribution of numerical features
numerical_features = ['area', 'bedrooms', 'bathrooms', 'stories', 'parking']
for feature in numerical_features:
    sns.histplot(data[feature], bins=20)
    plt.xlabel(feature)
    plt.ylabel('Frequency')
    plt.title(f'Distribution of {feature}')
    plt.show()


# In[183]:


# Explore the relationship between numerical features and the target variable
for feature in numerical_features:
    sns.scatterplot(x=feature, y='price', data=data)
    plt.xlabel(feature)
    plt.ylabel('Price')
    plt.title(f'{feature} vs. Price')
    plt.show()


# ## Step 2: Data Encoding

# In[184]:


data['mainroad'] = data['mainroad'].replace({'yes':1,'no':0})
data['guestroom'] = data['guestroom'].replace({'yes':1,'no':0})
data['basement'] = data['basement'].replace({'yes':1,'no':0})
data['hotwaterheating'] = data['hotwaterheating'].replace({'yes':1,'no':0})
data['airconditioning'] = data['airconditioning'].replace({'yes':1,'no':0})
data['furnishingstatus'] = data['furnishingstatus'].replace({'furnished':2,'semi-furnished':1,'unfurnished':0})
data['prefarea'] = data['prefarea'].replace({'yes':1,'no':0})


# In[185]:


data.head()


# ## Step 3: Feature Selection and Splitting

# In[186]:


from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
numerical_features = ['area', 'bedrooms', 'bathrooms', 'stories', 'parking', 'price_per_sqft']
data[numerical_features] = scaler.fit_transform(data[numerical_features])


# In[187]:


X = data.drop(columns=['price'])  # Features
y = data['price']  # Target variable


# In[188]:


X.head()


# In[190]:


X.shape


# In[155]:


# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# In[157]:


rfr = RandomForestRegressor()


# In[158]:


model = rfr.fit(X_train,y_train)


# In[159]:


y_pred_rfr = model.predict(X_test)


# In[160]:


r2_score(y_pred_rfr,y_test)


# In[161]:


import joblib


# In[164]:


joblib.dump(model, 'models/trained_model_1.joblib')

