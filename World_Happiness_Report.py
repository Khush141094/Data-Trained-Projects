#!/usr/bin/env python
# coding: utf-8

# In[28]:


import pandas as pd
import numpy as np

data = pd.read_csv(r'C:\Users\s0116731\Desktop\happiness_score_dataset.csv')
# data.head()
happiness = data.drop(['Country','Happiness Rank','Region'],axis=1)

# happiness.head()

from sklearn.model_selection import train_test_split
X = happiness[['Economy (GDP per Capita)','Family','Health (Life Expectancy)','Freedom','Trust (Government Corruption)','Generosity']]
y = happiness['Happiness Score']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

from sklearn.preprocessing import StandardScaler
scale = StandardScaler()
happiness = scale.fit_transform(happiness)

from sklearn.linear_model import LinearRegression
linear_reg = LinearRegression()
linear_reg.fit(X_train, y_train)

y_predict = linear_reg.predict(X_test)
pd.DataFrame({'Predicted': y_predict})


# In[30]:


pd.DataFrame({'Actual': y_test, 'Predicted': y_predict})


# In[ ]:




