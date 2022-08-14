#!/usr/bin/env python
# coding: utf-8

# # Problem statement
# 

# The introduction of online payment systems has helped a lot in the ease of payments. But, at the same time, it increased in payment frauds. Online payment frauds can happen with anyone using any payment system, especially while making payments using a credit card. That is why detecting online payment fraud is very important for credit card companies to ensure that the customers are not getting charged for the products and services they never paid.

# To identify online payment fraud with machine learning, we need to train a machine learning model for classifying fraudulent and non-fraudulent payments.
#                             from the data  we have analysed that what type of transactions lead to fraud.

# # 2.understandanding the data with stats 

# While working with machine learning projects, usually we ignore two most important parts called mathematics and data. It is because, we know that ML is a data driven approach and our ML model will produce only as good or as bad results as the data we provided to it. 

# # Looking at Raw Data

#  It is important to look at raw data because the insight we will get after looking at raw data will boost our chances to better pre-processing as well as handling of data for ML projects.

# In[5]:


import pandas as pd
import pandas as pd

import numpy as np


# In[6]:


df=pd.read_csv(r"C:\Users\admin\Downloads\online.csv")


# In[7]:


df.head()


# By looking at raw data we gathered information such as-
# step: represents a unit of time where 1 step equals 1 hour
# type: type of online transaction
# amount: the amount of the transaction
# nameOrig: customer starting the transaction
# oldbalanceOrg: balance before the transaction
# newbalanceOrig: balance after the transaction
# nameDest: recipient of the transaction
# oldbalanceDest: initial balance of recipient before the transaction
# newbalanceDest: the new balance of recipient after the transaction
# 
# Each feature have a significant role in our model

# # Checking Dimensions of Data 

# It is always a good practice to know how much data, in terms of rows and columns, we are having for our ML project. The reasons behind are: 
#  Suppose if we have too many rows and columns then it would take long time to run the algorithm and train the model. 
#  
#  Suppose if we have too less rows and columns then it we would not have enough data to well train the model. 

# In[9]:


df.shape


# We can easily observe from the output that our dataset, we are going to use, is having 6362620 rows and 11 columns.

# # Getting Each Attribute’s Data Type 

# It is another good practice to know data type of each attribute. The reason behind is that, as per to the requirement, sometimes we may need to convert one data type to another. For example, we may need to convert string into floating point or int for representing categorial or ordinal values. We can have an idea about the attribute’s data type by looking at the raw data, but another way is to use dtypes property of Pandas DataFrame. 

# In[10]:


df.dtypes


# From the above output, we can easily get the datatypes of each attribute. 

# # Statistical Summary of Data 

# We have discussed Python recipe to get the shape i.e. number of rows and columns, of data but many times we need to review the summaries out of that shape of data. It can be done with the help of describe() function of Pandas DataFrame that further provide the following 8 statistical properties of each & every data attribute: 
#  Count  Mean  Standard Deviation   Minimum Value  Maximum value  25%  Median i.e. 50%  75% 

# In[11]:


df.describe()


# # Reviewing Class Distribution 

# Class distribution statistics is useful in classification problems where we need to know the balance of class values. It is important to know class value distribution because if we have highly imbalanced class distribution i.e. one class is having lots more observations than other class, then it may need special handling at data preparation stage of our ML project. We can easily get class distribution in Python with the help of Pandas DataFrame

# In[14]:


df['isFraud'].unique()


# In[17]:


count_class = df.groupby('isFraud').size() 
print(count_class) 


# From the above output, it can be clearly seen that the number of observations with class 0 are greater than number of observations with class 1.   

# In[ ]:





# In[4]:


#Now, let’s have a look at whether this dataset has any null values or not:
df.isnull().sum()


# In[ ]:





# In[5]:


df.info()


# So this dataset does not have any null values. Before moving forward, now, let’s have a look at the type of transaction mentioned in the dataset:

# In[6]:


print(df.type.value_counts())


# In[7]:


df.size


# In[8]:


df.columns


# In[9]:


import numpy as np
import matplotlib.pyplot as plt
# if using a Jupyter notebook, include:
get_ipython().run_line_magic('matplotlib', 'inline')


# In[10]:


type = df["type"].value_counts()
transactions = type.index
quantity = type.values


figure = plt.pie(type, 
             labels=transactions
             )


# # Reviewing Correlation between Attributes 

# The relationship between two variables is called correlation. In statistics, the most common method for calculating correlation is Pearson’s Correlation Coefficient. It can have three values as follows: 
#  Coefficient value = 1: It represents full positive correlation between variables.  Coefficient value = -1: It represents full negative correlation between variables.  Coefficient value = 0: It represents no correlation at all between variables. 

# Now let’s have a look at the correlation between the features of the data with the isFraud column:

# In[ ]:





# In[11]:


correlation = df.corr()
print(correlation["isFraud"].sort_values(ascending=False))


# # Reviewing Skew of Attribute Distribution 

# Skewness may be defined as the distribution that is assumed to be Gaussian but appears distorted or shifted in one direction or another, or either to the left or right. Reviewing the skewness of attributes is one of the important tasks due to following reasons:  
#  Presence of skewness in data requires the correction at data preparation stage so that we can get more accuracy from our model.  
#  
#  Most of the ML algorithms assumes that data has a Gaussian distribution i.e. either normal of bell curved data

# In[18]:


df.skew()


# From the above output, positive or negative skew can be observed. If the value is closer to zero, then it shows less skew. 

# # Understanding Data with Visualization 

# With the help of data visualization, we can see how the data looks like and what kind of correlation is held by the attributes of data. It is the fastest way to see if the features correspond to the output. With the help of following Python recipes, we can understand ML data with statistics. 

# # Univariate Plots: Understanding Attributes Independently 

# The simplest type of visualization is single-variable or “univariate” visualization. With the help of univariate visualization, we can understand each attribute of our dataset independently. The following are some techniques in Python to implement univariate visualization

# # Histograms 

# Histograms group the data in bins and is the fastest way to get idea about the distribution of each attribute in dataset. The following are some of the characteristics of histograms:  It provides us a count of the number of observations in each bin created for visualization.

# In[ ]:


 From the shape of the bin, we can easily observe the distribution i.e. weather it is Gaussian, skewed or exponential. 
 
 Histograms also help us to see possible outliers


# In[19]:


from matplotlib import pyplot 


# In[20]:


df.hist() 
pyplot.show() 


# The above output shows that it created the histogram for each attribute in the dataset.

# # Density Plots 

# Another quick and easy technique for getting each attributes distribution is Density plots. It is also like histogram but having a smooth curve drawn through the top of each bin. We can call them as abstracted histograms

# # Box and Whisker Plots 

# Box and Whisker plots, also called boxplots in short, is another useful technique to review the distribution of each attribute’s distribution. The following are the characteristics of this technique: 
#  It is univariate in nature and summarizes the distribution of each attribute.  
#  
#  It draws a line for the middle value i.e. for median. 
#  
#  It draws a box around the 25% and 75%. 
#  
#  It also draws whiskers which will give us an idea about the spread of the data. 
#  
#  The dots outside the whiskers signifies the outlier values. Outlier values would be 1.5 times greater than the size of the spread of the middle data. 

# In[22]:


df.plot(kind='box', subplots=True, layout=(3,3)) 
pyplot.show() 


# # Now let’s transform the categorical features into numerical. Here I will also transform the values of the isFraud column into No Fraud and Fraud labels to have a better understanding of the output:

# In[24]:


df["type"] = df["type"].map({"CASH_OUT": 1, "PAYMENT": 2, 
                                 "CASH_IN": 3, "TRANSFER": 4,
                                 "DEBIT": 5})
df["isFraud"] = df["isFraud"].map({0: "No Fraud", 1: "Fraud"})


# In[25]:


df.head()


# # Online Payments Fraud Detection Model
# Now let’s train a classification model to classify fraud and non-fraud transactions. Before training the model, I will split the data into training and test sets:
# 
# 

# In[38]:


# splitting the data
from sklearn.model_selection import train_test_split
x = np.array(df[["type", "amount", "oldbalanceOrg", "newbalanceOrig"]])
y = np.array(df[["isFraud"]])


# Now let’s train the online payments fraud detection model:

# In[47]:


# training a machine learning model
from sklearn.tree import DecisionTreeClassifier
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25, random_state=42)
model = DecisionTreeClassifier()
model.fit(x_train, y_train)
print(model.score(x_test, y_test))


# Now let’s classify whether a transaction is a fraud or not by feeding about a transaction into the model:

# In[19]:


# prediction
#features = [type, amount, oldbalanceOrg, newbalanceOrig]
features = np.array([[4, 9000.60, 9000.60, 0.0]])
print(model.predict(features))


# # Summary
# So this is how we can detect online payments fraud with machine learning using Python. Detecting online payment frauds is one of the applications of data science in finance. I hope you liked this article on online payments fraud detection with machine learning using Python. Feel free to ask valuable questions in the comments section below.

# In[ ]:




