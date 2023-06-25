#!/usr/bin/env python
# coding: utf-8

# ## Scope:
# ## ●	Exploratory data analysis
# ## ●	Data Pre-processing
# ## ●	Training linear regression model with OLS method for prediction
# ## ●	Tuning the model to improve the performance
# 

# In[1]:


import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


# In[2]:


os.chdir(r"C:\PGA32\Machine Learning\Linear Regression\Project")


# In[3]:


ins_Char=pd.read_csv("167541498631660ec2379f8793842aa5b168a_.csv")


# In[4]:


ins_Char.head()


# In[5]:


ins_Char.info()


# In[6]:


ins_Char.isnull().sum()


# In[7]:


ins_Char.columns


# # Exploratory data analysis

# ## EDA

# ## Univeriate Analysis

# ### Categorical Variable

# In[8]:


ins_Char.dtypes[ins_Char.dtypes=="object"]


# In[9]:


ins_Char["sex"].nunique()


# In[10]:


ins_Char["sex"].unique()


# In[11]:


ins_Char["sex"].value_counts


# In[12]:


ins_Char["sex"].value_counts(normalize=True)


# In[13]:


ins_Char["sex"].value_counts().plot(kind="bar",figsize=(8,5))


# In[14]:


ins_Char["smoker"].nunique()


# In[15]:


ins_Char["smoker"].unique()


# In[16]:


ins_Char["smoker"].value_counts


# In[17]:


ins_Char["smoker"].value_counts().plot(kind="bar",figsize=(8,5))


# In[18]:


ins_Char["region"].nunique()


# In[19]:


ins_Char["region"].unique()


# In[20]:


ins_Char["region"].value_counts()


# In[21]:


ins_Char["region"].value_counts().plot(kind="bar",figsize=(8,5))


# ### Numerical Variable

# In[22]:


ins_Char.dtypes[ins_Char.dtypes!="object"]


# In[23]:


ins_Char["age"].min()


# In[24]:


ins_Char["age"].max()


# In[25]:


ins_Char["age"].mean()


# In[26]:


ins_Char["age"].skew()


# In[27]:


ins_Char["age"].std()


# In[28]:


plt.hist(ins_Char["age"])
plt.show()


# In[29]:


sns.histplot(ins_Char["age"])
plt.show()


# In[30]:


sns.distplot(ins_Char["age"])
plt.show()


# In[31]:


ins_Char["bmi"].min()


# In[32]:


ins_Char["bmi"].max()


# In[33]:


ins_Char["bmi"].mean()


# In[34]:


ins_Char["bmi"].std()


# In[35]:


ins_Char["bmi"].skew()


# In[36]:


plt.hist(ins_Char["age"])
plt.show()


# In[37]:


sns.histplot(ins_Char["bmi"])
plt.show()


# In[38]:


sns.distplot(ins_Char["bmi"])
plt.show()


# In[39]:


ins_Char["children"].min()


# In[40]:


ins_Char["children"].max()


# In[41]:


ins_Char["children"].mean()


# In[42]:


ins_Char["children"].std()


# In[43]:


ins_Char["children"].skew()


# In[44]:


plt.hist(ins_Char["children"])
plt.show()


# In[45]:


sns.histplot(ins_Char["children"])
plt.show()


# In[46]:


sns.distplot(ins_Char["children"])
plt.show()


# In[47]:


ins_Char["charges"].min()


# In[48]:


ins_Char["charges"].max()


# In[49]:


ins_Char["charges"].std()


# In[50]:


ins_Char["charges"].mean()


# In[51]:


ins_Char["charges"].skew()


# In[52]:


plt.hist(ins_Char["charges"])
plt.show()


# In[53]:


sns.histplot(ins_Char["charges"])
plt.show()


# In[54]:


sns.distplot(ins_Char["charges"])
plt.show()


# In[55]:


ins_Char.describe()


# In[56]:


ins_Char.describe(percentiles=[.01,.02,.03,.04,.05,.06,.07,.08,.09,.15,.75,.80,.86,.95,.45,.99])


# ## Bivariate Analysis

# In[57]:


ins_Char.dtypes[ins_Char.dtypes=="object"].index


# In[58]:


ins_Char.head()


# In[59]:


pd.crosstab(ins_Char["sex"], ins_Char["region"])


# In[60]:


pd.crosstab(ins_Char["sex"], ins_Char["region"],normalize="all")


# In[61]:


pd.crosstab(ins_Char["sex"], ins_Char["region"],normalize="index")


# In[62]:


pd.crosstab(ins_Char["sex"], ins_Char["region"],normalize="columns")


# ### Categorical to Numerical

# In[63]:


ins_Char.dtypes[ins_Char.dtypes!="object"].index


# In[64]:


ins_Char.groupby(["sex"]).agg({"age":["min", "max","mean", "std","count"]})


# In[65]:


ins_Char.groupby(["sex"]).agg({"bmi":["min", "max","mean", "std","count"]})


# In[66]:


ins_Char.groupby(["sex"]).agg({"children":["min", "max","mean", "std","count"]})


# In[67]:


ins_Char.groupby(["sex"]).agg({"charges":["min", "max","mean", "std","count"]})


# In[68]:


ins_Char[ins_Char["sex"]=="female"]


# In[69]:


ins_Char[ins_Char["sex"]=="male"]


# In[70]:


ins_Char.groupby(["region"]).agg({"age":["min", "max","mean", "std","count"]})


# In[71]:


ins_Char.groupby(["region"]).agg({"bmi":["min", "max","mean", "std","count"]})


# In[72]:


ins_Char.groupby(["region"]).agg({"children":["min", "max","mean", "std","count"]})


# In[73]:


ins_Char.groupby(["region"]).agg({"charges":["min", "max","mean", "std","count"]})


# In[74]:


ins_Char[ins_Char["region"]=="northeast"]


# In[75]:


ins_Char[ins_Char["region"]=="northwest"]


# In[76]:


ins_Char[ins_Char["region"]=="southeast"]


# In[77]:


ins_Char[ins_Char["region"]=="southwest"]


# In[78]:


ins_Char.groupby(["region"]).agg({"charges":["min"]}).plot(kind="bar")


# ### Numerical to Numerical

# In[79]:


plt.scatter(ins_Char["sex"], ins_Char["age"])
plt.xlabel("Matual")
plt.ylabel("Old")
plt.title("Matual Vs Old")
plt.show()


# In[80]:


ins_Char[["sex", "age"]].corr()


# In[81]:


ins_Char.head()


# In[82]:


plt.scatter(ins_Char["bmi"], ins_Char["charges"])
plt.xlabel("bmi_value")
plt.ylabel("Tax")
plt.title("bmi_value Vs Tax")
plt.show()


# In[83]:


sns.pairplot(ins_Char)
plt.show()


# In[85]:


ins_Char.plot(figsize=(12,6))


# # Data Pre-processing

# ## Missing Value treatment

# In[86]:


ins_Char.isnull().sum()


# In[87]:


ins_Char.shape


# ## Outlier Treatment

# In[88]:


ins_Char.columns


# In[89]:


var="bmi"
q1=ins_Char[var].quantile(.25)
q3=ins_Char[var].quantile(.75)
iqr=q3-q1
upper_bound=q3+1.5*iqr
lower_bound=q1-1.5*iqr
ins_Char[var]=np.where(ins_Char[var]>=upper_bound, upper_bound,ins_Char[var])


# In[90]:


upper_bound


# In[91]:


lower_bound


# In[92]:


var="charges"
q1=ins_Char[var].quantile(.25)
q3=ins_Char[var].quantile(.75)
iqr=q3-q1
upper_bound=q3+1.5*iqr
lower_bound=q1-1.5*iqr
ins_Char[var]=np.where(ins_Char[var]>=upper_bound, upper_bound,ins_Char[var])


# In[93]:


upper_bound


# In[94]:


lower_bound


# In[96]:


ins_Char.dtypes[ins_Char.dtypes!="object"].index


# In[97]:


var="age"
q1=ins_Char[var].quantile(.25)
q3=ins_Char[var].quantile(.75)
iqr=q3-q1
upper_bound=q3+1.5*iqr
lower_bound=q1-1.5*iqr
ins_Char[var]=np.where(ins_Char[var]>=upper_bound, upper_bound,ins_Char[var])


# In[98]:


upper_bound


# In[99]:


lower_bound


# In[100]:


var="children"
q1=ins_Char[var].quantile(.25)
q3=ins_Char[var].quantile(.75)
iqr=q3-q1
upper_bound=q3+1.5*iqr
lower_bound=q1-1.5*iqr
ins_Char[var]=np.where(ins_Char[var]>=upper_bound, upper_bound,ins_Char[var])


# In[101]:


upper_bound


# In[103]:


lower_bound


# In[104]:


sns.boxplot(ins_Char["bmi"])
plt.show()


# In[106]:


sns.boxplot(ins_Char["charges"])
plt.show()


# # ●	Training linear regression model with OLS method for prediction

# ### Model development Stage

# In[107]:


ins_Char.columns


# In[108]:


X = ins_Char[[ 'age',
       'bmi',
       'children'
      ]]
Y = ins_Char["charges"]


# In[109]:


import statsmodels.api as sm  # Statical Learning
X = sm.add_constant(X)


# In[110]:


model = sm.OLS(Y, X).fit()


# In[111]:


print(model.summary())


# In[112]:


from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.linear_model import LinearRegression


# In[113]:


y=ins_Char["charges"]
x=ins_Char[[ 'age',
       'bmi',
       'children'
      ]]


# In[114]:


x_train, x_test, y_train, y_test = train_test_split(x,y, test_size=.3, random_state=88)


# In[115]:


x_train.head()


# In[116]:


x_train.shape


# In[117]:


x_test.head()


# In[118]:


x_test.shape


# In[119]:


linReg=LinearRegression()


# In[120]:


linReg.fit(x_train, y_train)


# In[121]:


linReg.coef_


# In[122]:


linReg.intercept_


# ### Evaluation of Model

# In[123]:


train_predict=linReg.predict(x_train)
train_predict


# In[124]:


test_predict=linReg.predict(x_test)
test_predict


# In[125]:


print("MSE of Train", np.mean((train_predict - y_train)**2))
print("RMSE of Train", np.sqrt(np.mean((train_predict-y_train)**2)))

print("MAE", np.mean(np.abs(train_predict- y_train)))
print("MAPE", np.mean(np.abs((train_predict-y_train)/y_train)))


# In[126]:


print("MSE of Test", np.mean((test_predict - y_test)**2))
print("RMSE of Test", np.sqrt(np.mean((test_predict - y_test)**2)))

print("MAE", np.mean(np.abs(test_predict- y_test)))
print("MAPE", np.mean(np.abs((test_predict-y_test)/y_test)))


# In[127]:


print("MSE of Train", np.mean((train_predict - y_train)**2))
print("RMSE of Train", np.sqrt((train_predict-y_train)**2))

print("MAE", np.mean(np.abs(train_predict- y_train)))
print("MAPE", np.mean(np.abs((train_predict-y_train)/y_train)))


# In[128]:


print("MSE of Test", np.mean((test_predict - y_test)**2))
print("RMSE of Test", np.sqrt((test_predict - y_test)**2))

print("MAE", np.mean(np.abs(test_predict- y_test)))
print("MAPE", np.mean(np.abs((test_predict-y_test)/y_test)))


# # Tuning the model to improve the performance
# 
# 

# In[140]:


from sklearn.linear_model import Ridge, RidgeCV, Lasso


# ### L1= Lasso Regression and L2= Ridge Regression

# In[144]:


# Lasso Regression Model
lr = LinearRegression()


lr.fit(x_train, y_train)

actual = y_test

train_score_lr = lr.score(x_train, y_train)
test_score_lr = lr.score(x_test, y_test)

print("The train score for lr model is {}".format(train_score_lr))
print("The test score for lr model is {}".format(test_score_lr))


# In[145]:


#Ridge Regression Model
linReg = Ridge(alpha=10)

linReg.fit(x_train,y_train)


train_score_ridge = linReg.score(x_train, y_train)
test_score_ridge = linReg.score(x_test, y_test)

print("The train score for ridge model is {}".format(train_score_ridge))
print("The test score for ridge model is {}".format(test_score_ridge))


# In[ ]:




