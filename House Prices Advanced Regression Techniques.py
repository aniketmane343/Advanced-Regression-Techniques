#!/usr/bin/env python
# coding: utf-8

# # House Prices Prediction Advanced Regression Techniques

# Import Libararies

# In[1]:


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from statsmodels.stats.outliers_influence import variance_inflation_factor
import warnings
from sklearn import metrics
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
# Import  LinearRegression,Lasso,Ridge,ElasticNet regression from sklearn
from sklearn.linear_model import LinearRegression,Lasso, Ridge, ElasticNetCV
from sklearn.preprocessing import PolynomialFeatures
from sklearn import metrics
from sklearn.model_selection import cross_val_score
from sklearn.metrics import mean_squared_error,r2_score

warnings.filterwarnings("ignore")


# Display data

# In[2]:


train=pd.read_csv("D:\\new project\\house-prices-advanced-regression-techniques\\train.csv")
train.head()


# In[3]:


##print shape of the data
train.shape


# From the above data some of the feature id is not required

# In[4]:


print("Id of Houses {}",format(len(train.Id)))
train=train.drop(columns="Id")
train.head()


# Data Discription

# In[5]:


with open("D:\\new project\\house-prices-advanced-regression-techniques\\data_description.txt") as f:
    train_decription=print(f.read())


# Finding the unique values in each column (type object)

# In[6]:


for col in train.select_dtypes('O').columns:
    print('We have {} unique values in {} column : {}'.format(len(train[col].unique()),col,train[col].unique()))
    print('-'*100)


# In[7]:


## histogram
hist=sns.distplot(train['SalePrice']);
hist


# skewness and kurtosis

# In[8]:


print('skewness:%f'% train['SalePrice'].skew())
print('kurtosis:%f'% train['SalePrice'].kurt())


# Correlation of the Train data

# In[9]:


train.corr()['SalePrice'].sort_values()


# In[10]:


corr=train.corr()
plt.figure(figsize=(12, 10))
sns.heatmap(corr[(corr >= 0.4) | (corr <= -0.4)], 
            cmap='viridis', vmax=1.0, vmin=-1.0, linewidths=0.1,
            annot=True, annot_kws={"size": 8}, square=True);


# In[11]:


train.info()


# In[12]:


#How many columns with different datatypes are there?
train.dtypes.value_counts()


# ## To deal with missing values

# In[13]:


train.isnull().sum(axis=0)


# In[14]:


def missing_percent(train):
    nan_percent= 100*(train.isnull().sum()/len(train))
    nan_percent= nan_percent[nan_percent>0].sort_values()
    return nan_percent
nan_percent=missing_percent(train)
nan_percent


# In[15]:


# Let's visualize the percentage of missing data on a graph
plt.figure(figsize=(12,6))
sns.barplot(x=nan_percent.index, y=nan_percent)
plt.xticks(rotation=90)


# And now let's look at the ones with less than 1%.

# In[16]:


nan_percent[nan_percent<1]


# In[17]:


train["Electrical"].unique()


# In[18]:


train[train['Electrical'].isnull()][['Electrical']]


# If we look at the data information: Electrical: Electrical system
# Electrical: Electrical system
# 
#        SBrkr	Standard Circuit Breakers & Romex
#        FuseA	Fuse Box over 60 AMP and all Romex wiring (Average)	
#        FuseF	60 AMP Fuse Box and mostly Romex wiring (Fair)
#        FuseP	60 AMP Fuse Box and mostly knob & tube wiring (poor)
#        Mix	Mixed
#      
#  It is just one row so we drop it.

# In[19]:


train[train['MasVnrType'].isnull()][['MasVnrType']]


# In[20]:


train["MasVnrType"].unique()


# If we look at the data information we can see:MasVnrType: Masonry veneer type
# 
#        BrkCmn	Brick Common
#        BrkFace	Brick Face
#        CBlock	Cinder Block
#        None	None
#        Stone	Stone
#        
#  So the ones with null value have no Masonry veneer and this is not a missing data. We should replace it with None.

# In[21]:


train[train['MasVnrArea'].isnull()][['MasVnrArea']]


# If we look at the data information we can see:
# MasVnrArea: Masonry veneer area in square feet
# 
# We have obviosly a low rate of missing values here, so we also drop them.

# In[22]:


train=train.dropna(axis=0,subset=['MasVnrArea','MasVnrType','Electrical'])


# In[23]:


missing_percent(train)

Basement data: BsmtQual, BsmtCond, BsmtFinType1, BsmtExposure, BsmtFinType2
Let's look at the information about the basement:

BsmtQual: Evaluates the height of the basement

   Ex   Excellent (100+ inches) 
   Gd   Good (90-99 inches)
   TA   Typical (80-89 inches)
   Fa   Fair (70-79 inches)
   Po   Poor (<70 inches
   NA   No Basement

BsmtCond: Evaluates the general condition of the basement

   Ex   Excellent
   Gd   Good
   TA   Typical - slight dampness allowed
   Fa   Fair - dampness or some cracking or settling
   Po   Poor - Severe cracking, settling, or wetness
   NA   No Basement

BsmtFinType1: Rating of basement finished area

   GLQ  Good Living Quarters
   ALQ  Average Living Quarters
   BLQ  Below Average Living Quarters   
   Rec  Average Rec Room
   LwQ  Low Quality
   Unf  Unfinshed
   NA   No Basement

BsmtExposure: Refers to walkout or garden level walls

   Gd   Good Exposure
   Av   Average Exposure (split levels or foyers typically score average or above)  
   Mn   Mimimum Exposure
   No   No Exposure
   NA   No Basement

BsmtFinType2: Rating of basement finished area (if multiple types)

   GLQ  Good Living Quarters
   ALQ  Average Living Quarters
   BLQ  Below Average Living Quarters   
   Rec  Average Rec Room
   LwQ  Low Quality
   Unf  Unfinshed
   NA   No Basement
So the Na values are not missing data, the house simply has no basement. We have to change them to None
# In[24]:


train['BsmtQual']= train['BsmtQual'].fillna('None')
train['BsmtCond']= train['BsmtCond'].fillna('None')
train['BsmtFinType1']= train['BsmtFinType1'].fillna('None')
train['BsmtExposure']= train['BsmtExposure'].fillna('None')
train['BsmtFinType2']= train['BsmtFinType2'].fillna('None')


# In[25]:


missing_percent(train)

Garage data: GarageCond, GarageQual, GarageFinish, GarageType, GarageYrBlt
Let's look at the information about the garage:

GarageType: Garage location

   2Types   More than one type of garage
   Attchd   Attached to home
   Basment  Basement Garage
   BuiltIn  Built-In (Garage part of house - typically has room above garage)
   CarPort  Car Port
   Detchd   Detached from home
   NA   No Garage

GarageYrBlt: Year garage was built

GarageFinish: Interior finish of the garage

   Fin  Finished
   RFn  Rough Finished  
   Unf  Unfinished
   NA   No Garage

GarageQual: Garage quality

   Ex   Excellent
   Gd   Good
   TA   Typical/Average
   Fa   Fair
   Po   Poor
   NA   No Garage

GarageCond: Garage condition

   Ex   Excellent
   Gd   Good
   TA   Typical/Average
   Fa   Fair
   Po   Poor
   NA   No Garage
So the Na values are not missing data, the house simply has no garage. We have to change them to None.
# In[26]:


train['GarageType']= train['GarageType'].fillna('None')
train['GarageFinish']= train['GarageFinish'].fillna('None')
train['GarageQual']= train['GarageQual'].fillna('None')
train['GarageCond']=train['GarageCond'].fillna('None')

GarageYrBlt: Year garage was built
Here we replace GarageYrBlt with mode value.
# In[27]:


train["GarageYrBlt"].mode()


# In[28]:


train['GarageYrBlt'].unique()


# In[29]:


train['GarageYrBlt']=train['GarageYrBlt']. fillna(train['GarageYrBlt']. mode()[0])


# In[30]:


missing_percent(train)


# In this missing_percent data more than 80% missing values are present so we drop this variable
# More than 80%: Fence, Alley, Miscfeature, PoolQC

# In[31]:


train=train.drop(columns=['Fence','Alley','MiscFeature','PoolQC'])


# In[32]:


missing_percent(train)

As we can see from the information: FireplaceQu: Fireplace quality

   Ex   Excellent - Exceptional Masonry Fireplace
   Gd   Good - Masonry Fireplace in main level
   TA   Average - Prefabricated Fireplace in main living area or Masonry Fireplace in basement
   Fa   Fair - Prefabricated Fireplace in basement
   Po   Poor - Ben Franklin Stove
   NA   No Fireplace
So the Na values are not missing data, the house simply has no fireplace. We have to change them to None.
# In[33]:


train['FireplaceQu']= train['FireplaceQu'].fillna('None')


# In[34]:


missing_percent(train)

LotFrontage: Linear feet of street connected to property
    
       
       We have 17.7% of missing data and we cannot just drop it. Lot frontage means the side of a lot abutting on a legally accessible street right-of-way other than an alley or an improved county road. We need to replace the Nan value with a suitable amount. Let's look if the lotfronatage summary statistics and unique values.
       
# In[35]:


train['LotFrontage'].describe()


# In[36]:


## histogram
hist=sns.distplot(train['LotFrontage']);
hist


# In[37]:


#skewness and kurtosis
print('skewness:%f'% train['LotFrontage'].skew())
print('kurtosis:%f'% train['LotFrontage'].kurt())


# In[38]:


train["LotFrontage"].fillna(train["LotFrontage"].mean(),inplace = True)


# In[39]:


missing_percent(train)


# In[40]:


train


# Finally we don't have any missing data!

# ## To Deal with Outlier Value

# list of variables that contain year information

# In[41]:


numerical_feature_train= train.select_dtypes(exclude='object')
categorical_feature_train= train.select_dtypes(include='object')


# Visualising numerical variables with Target Variables

# In[42]:


fig,axs= plt.subplots(12,3,figsize=(20,80))
fig.subplots_adjust(hspace=0.6)
for i,ax in zip(numerical_feature_train.columns,axs.flatten()):
    sns.scatterplot(x=i, y=train['SalePrice'], hue=train['SalePrice'],data=numerical_feature_train,ax=ax,palette='viridis_r')
    plt.xlabel(i,fontsize=12)
    plt.ylabel('SalePrice',fontsize=12)
    #ax.set_yticks(np.arange(0,900001,100000))
    ax.set_title('SalePrice'+' - '+str(i),fontweight='bold',size=20)


# from above graphs we can see the varibles LotFrontage, LotArea, MasVnrArea, BsmtFinsf1, TotalBsmtsf, 1stFlrsf, 2ndFlrsf, GrLivArea, GrageArea this variables have high outliers. means extremely large areas for very low prices. so we replace these outliers by its lower values.

# In[43]:


train.LotFrontage[(train.LotFrontage >= 160)] = 160
train.LotArea[(train.LotArea >= 75000)] = 75000
train.MasVnrArea[(train.MasVnrArea >= 1000)] = 1000
train.BsmtFinSF1[(train.BsmtFinSF1 >= 2500)] = 2500
train.TotalBsmtSF[(train.TotalBsmtSF >= 3000)] = 3000
train['1stFlrSF'][(train['1stFlrSF'] >= 3000)] = 3000
train.GrLivArea[(train.GrLivArea >= 3500)] = 3500
train.GarageArea[(train.GarageArea >= 1500)] = 1500


# Visualising Categorical predictor variables with Target Variables

# In[44]:


def facetgrid_boxplot(x, y, **kwargs):
    sns.boxplot(x=x, y=y)
    x=plt.xticks(rotation=90)
    

f = pd.melt(train, id_vars=['SalePrice'], value_vars=sorted(train[categorical_feature_train.columns]))
g = sns.FacetGrid(f, col="variable", col_wrap=3, sharex=False, sharey=False, size=5)
g = g.map(facetgrid_boxplot, "value", "SalePrice")


#    ## Labeleling Categorical Feature

# In[45]:


categorical_feature_train.head(2)


# In[46]:


for c in categorical_feature_train:
    lbl = LabelEncoder() 
    lbl.fit(list(categorical_feature_train[c].values)) 
    categorical_feature_train[c] = lbl.transform(list(categorical_feature_train[c].values))

# shape        
print('Shape data: {}'.format(categorical_feature_train.shape))


# In[47]:


categorical_feature_train.head(2)


# In[48]:


Y=train["SalePrice"]
Y


# In[49]:


numerical_feature_train= numerical_feature_train.drop(columns="SalePrice")


# In[50]:


X=pd.concat([categorical_feature_train,numerical_feature_train], axis=1)
X.head()


# ## Model Fitting

# In[51]:


x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.30, random_state=42)


# In[59]:


print(x_train.shape)
print(x_test.shape)
print(y_train.shape)
print(y_test.shape)


# In[54]:


X.columns


#  ## Linear  Regression Model

# In[55]:


Linear_reg = LinearRegression()
Linear_reg.fit(x_train,y_train)


# In[56]:


coeff_df = pd.DataFrame(Linear_reg.coef_, X.columns, columns=['Coefficient'])
coeff_df


# In[60]:


Linear_reg.intercept_


# In[61]:


y_pred_Linear=Linear_reg.predict(x_test)
pd.DataFrame(y_pred_Linear,columns=["Predicted_Value"])


# In[62]:


R_squared=metrics.r2_score(y_test,y_pred_Linear)
MAE_linear=metrics.mean_absolute_error(y_test , y_pred_Linear)
MSE_linear=metrics.mean_squared_error(y_test , y_pred_Linear)
RMSE_linear=np.sqrt(MSE_linear)
Quantity=pd.DataFrame([R_squared,MAE_linear,MSE_linear,RMSE_linear], index=['R_squared','MAE_linear','MSE_linear','RMSE_linear'],columns=['Quantity'])
Quantity


# In[63]:


Accuaracy_LinearRegression=r2_score(y_test,y_pred_Linear)
print("Accuracy of Linear Regression Model is",format(Accuaracy_LinearRegression))


# In[64]:


result = Linear_reg.predict(x_train)
compare_df = pd.DataFrame()
compare_df['Actual Values'] = y_train
compare_df['Predicted Values'] = result
compare_df


# In[65]:


plt.figure(figsize = (15,8))
plt.plot(y_train,'o', color = 'blue',label = 'Actual Values')
plt.plot(result,'*', color = 'red',label = 'Predicted values')


# ## Lasso Regression  Model

# In[79]:


Lasso_reg = Lasso()
Lasso_reg.fit(x_train,y_train)


# In[80]:


coeff_df = pd.DataFrame(Lasso_reg.coef_, X.columns, columns=['Coefficient'])
coeff_df.head(2)


# In[81]:


y_pred_Lasso=Lasso_reg.predict(x_test)
pd.DataFrame(y_pred_Lasso,columns=["Predicted_Value"])


# In[82]:


result = Lasso_reg.predict(x_train)
compare_df = pd.DataFrame()
compare_df['Actual Values'] = y_train
compare_df['Predicted Values'] = result
compare_df


# In[83]:


print("R squared",metrics.r2_score(y_test,y_pred_Lasso))


# In[84]:


R_squared=metrics.r2_score(y_test,y_pred_Lasso)
MAE_Lasso=metrics.mean_absolute_error(y_test , y_pred_Lasso)
MSE_Lasso=metrics.mean_squared_error(y_test , y_pred_Lasso)
RMSE_Lasso=np.sqrt(MSE_Lasso)
Quantity=pd.DataFrame([R_squared,MAE_Lasso,MSE_Lasso,RMSE_Lasso], index=['R_squared','MAE_Lasso','MSE_Lasso','RMSE_Lasso'],columns=['Quantity'])
Quantity


# In[123]:


Accuaracy_LassoRegression=r2_score(y_test,y_pred_Lasso)
print("Accuracy of Lasso Regression Model is",Accuaracy_LassoRegression)


# In[86]:


plt.figure(figsize = (15,8))
plt.plot(y_train,'o', color = 'blue',label = 'Actual Values')
plt.plot(result,'*', color = 'Yellow',label = 'Predicted values')


# ## Ridge Regression Model

# In[87]:


Ridge_reg=Ridge()
Ridge_reg.fit(x_train,y_train)


# In[88]:


y_pred_Ridge=Ridge_reg.predict(x_test)
pd.DataFrame(y_pred_Ridge,columns=["Predicted_Value"])


# In[89]:


print("R squared",metrics.r2_score(y_test,y_pred_Ridge))


# In[90]:


R_squared=metrics.r2_score(y_test,y_pred_Ridge)
MAE_Ridge=metrics.mean_absolute_error(y_test , y_pred_Ridge)
MSE_Ridge=metrics.mean_squared_error(y_test , y_pred_Ridge)
RMSE_Ridge=np.sqrt(MSE_Ridge)
Quantity=pd.DataFrame([R_squared,MAE_Ridge,MSE_Ridge,RMSE_Ridge], index=['R_squared','MAE_Ridge','MSE_Ridge','RMSE_Ridge'],columns=['Quantity'])
Quantity


# In[116]:


Accuaracy_RidgeRegression=r2_score(y_test,y_pred_Ridge)
print("Accuracy of Ridge Regression Model is",format(Accuaracy_RidgeRegression))


# In[92]:


result = Ridge_reg.predict(x_train)
compare_df = pd.DataFrame()
compare_df['Actual Values'] = y_train
compare_df['Predicted Values'] = result
compare_df


# In[93]:


plt.figure(figsize = (15,8))
plt.plot(y_train,'o', color = 'blue',label = 'Actual Values')
plt.plot(result,'*', color = 'green',label = 'Predicted values')


# ## Elastic Net Regression Model

# In[94]:


Elastic_Regression = ElasticNetCV(l1_ratio=[.1, .5, .7,.9, .95, .99, 1], cv=5, max_iter=100000)
Elastic_Regression.fit(x_train, y_train)


# In[95]:


y_pred_elastic = Elastic_Regression.predict(x_test)
MAE =metrics.mean_absolute_error(y_test,y_pred_elastic)
MSE = mean_squared_error(y_test, y_pred_elastic)
RMSE = np.sqrt(MSE)


# In[96]:


pd.DataFrame({'Ridge Metrics':[MAE,MSE,RMSE]},index=['MAE','MSE','RMSE'])


# In[121]:


Accuaracy_ElasticnetRegression=r2_score(y_test,y_pred_elastic)
print("Accuracy of Elasticnet Regression Model is",Accuaracy_ElasticnetRegression)


# In[102]:


result = Elastic_Regression.predict(x_train)
compare_df = pd.DataFrame()
compare_df['Actual Values'] = y_train
compare_df['Predicted Values'] = result
compare_df


# In[103]:


plt.figure(figsize = (15,8))
plt.plot(y_train,'o', color = 'blue',label = 'Actual Values')
plt.plot(result,'*', color = 'orange',label = 'Predicted values')


# # Polynomial Regresssion Model

# In[104]:


poly_reg = PolynomialFeatures(degree =2)
X_poly = poly_reg.fit_transform(X)
polynomial_reg = LinearRegression()
polynomial_reg.fit(X_poly, Y)


# In[105]:


print(polynomial_reg.coef_)


# In[106]:


polynomial_reg.intercept_


# In[107]:


y_pred_poly = polynomial_reg.predict(poly_reg.transform(x_test))
pd.DataFrame(y_pred_poly,columns=["Predicted_Value"])


# In[108]:


print("R squared",metrics.r2_score(y_test,y_pred_poly))


# In[109]:


R_squared=metrics.r2_score(y_test,y_pred_poly)
MAE_poly=metrics.mean_absolute_error(y_test , y_pred_poly)
MSE_poly=metrics.mean_squared_error(y_test , y_pred_poly)
RMSE_poly=np.sqrt(MSE_poly)
Quantity=pd.DataFrame([R_squared,MAE_poly,MSE_poly,RMSE_poly], index=['R_squared','MAE_poly','MSE_poly','RMSE_poly'],columns=['Quantity'])
Quantity


# In[124]:


Accuaracy_PolynomialRegression=r2_score(y_test,y_pred_poly)
print("Accuracy of Polynomial Regression Model is",format(Accuaracy_PolynomialRegression))


# In[111]:


result = polynomial_reg.predict(X_poly)
compare_df = pd.DataFrame()
compare_df['Actual Values'] = Y
compare_df['Predicted Values'] = result
compare_df


# In[112]:


coeff_df = pd.DataFrame(polynomial_reg.coef_, columns=['Coefficient'])
coeff_df


# In[113]:


plt.figure(figsize = (15,8))
plt.plot(y_train,'o', color = 'blue',label = 'Actual Values')
plt.plot(result,'*', color = 'red',label = 'Predicted values')

Here, we see that the model have accuracy  0.9999999999999917 then we say that model is overfitted
# # Accuaracy Of The Model

# Accuracy is defined as the percentage of correct predictions for the test data. It can be calculated easily by dividing the number of correct predictions by the number of total predictions.
# 
#                 accuracy=correctpredictions/allpredictions
# 

# In[125]:


Accuracy=pd.DataFrame([Accuaracy_LinearRegression,Accuaracy_LassoRegression,Accuaracy_RidgeRegression,Accuaracy_ElasticnetRegression,Accuaracy_PolynomialRegression], index=['Accuaracy_LinearRegression','Accuaracy_LassoRegression','Accuaracy_RidgeRegression','Accuaracy_ElasticnetRegression','Accuaracy_PolynomialRegression'],columns=['Accuracy'])
Accuracy


# ## Conclusion

# Here We see that Accuracy of models LinearRegression,LassoRegression and RidgeRegression is 80%. we can use any model out of three for predication.
# In ElasticnetRegression model Accuracy is 74% we didn't use this model for prediction.And the Polynomial model have accuracy approximetly 100% so it has overfitting problem. so we cannot use this model for prediction

# In[ ]:




