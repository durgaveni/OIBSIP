#!/usr/bin/env python
# coding: utf-8

# ###DURGAVENI RAVIPATI | DATA SCIENCE INTERN | OASIS INFOBYTE | #TASK 3

#  CAR PRICE PREDICTION WITH MACHINE LEARNING

# In[1]:


# IMPORT NECESSARY LIBRARIES
#import numpy library as np
import numpy as np
#import pandas library as pd
import pandas as pd
#import matplotlib.pyplot as plt
import matplotlib.pyplot as plt
#import seaborn as sns
import seaborn as sns
#feature warnings
import warnings
warnings.filterwarnings('ignore')

from sklearn.preprocessing import LabelEncoder


# In[2]:


#Load car dataset:-
df = pd.read_csv("car data.csv")


# In[3]:


#first five rows of dataset:- 
df.head()


# In[4]:


#last five rows of dataset:-
df.tail()


# In[5]:


#shape of the dataset:-
df.shape


# In[6]:


#information of the dataset:-
df.info()


# In[7]:


#data types by each column
df.dtypes


# In[8]:


#describe:-
df.describe(include='object')


# HANDLING MISSING DATA

# In[9]:


#check null values:-
df.isnull().sum()


# There is no null values present in the dataset.

# In[10]:


#check duplicate values:-
df.duplicated().sum()


# In[11]:


#drop the duplicated values
df.drop_duplicates(inplace = True)


# In[12]:


df.duplicated().sum()


# There is no duplicate values pressent in the dataset.

# In[13]:


df.columns=['Car_Name', 'Year', 'Selling_Price', 'Present_Price', 'Driven_kms',
       'Fuel_Type', 'Selling_type', 'Transmission', 'Owner']


# In[14]:


df.head(10)


# In[15]:


#correlation:-
corr =df.corr()
corr


# In[16]:


#plot a heatmap for the correlation matrix
#annot : print values in each cell
#Linewidths: specify width of the line and specifying the plot
#cmap : colour code for the plot
#fmt : set the decimal place of the annot
sns.heatmap(corr, annot=True,linewidths=0.2, cmap='plasma', fmt = '.1g')
plt.title("Correlation Matrix", fontsize=15)
#display the plot :
plt.show()


# In[17]:


df['Transmission'].value_counts()


# EXPLORATORY DATA ANALYSIS

# PIE CHART

# In[18]:


#Visualize the piechart:-
car_names=df['Car_Name'].value_counts()
label=['city','corolla altis','verna','fortuner','brio','ciaz','innova','i20','grand i10','jazz','amaze','Royal Enfield Classic 350','ertiga','eon','sx4','alto k10','i10','swiift','Bajaj Pulsar 150','Royal Enfield Thunder 350']
plt.figure(figsize=(16,9))
plt.pie(car_names[:20],labels=label, autopct='%1.2f%%')
plt.title("to visualize the top 20 cars sold",fontsize=32,fontweight='bold')
plt.show()


# BAR PLOT

# In[19]:


#Visualize the barplot:-
year=df['Year'].value_counts()
ax=plt.axes()
ax.set(facecolor='yellow')
sns.set(rc={'figure.figsize':(20,8)},style='darkgrid')
ax.set_title("to visualize the which year most car sold",fontsize=20,fontweight=900)
sns.barplot(x=year.index,y=year,palette='rainbow')
plt.xlabel("Year")
plt.ylabel("count")
plt.show()


# DISTPLOT

# In[20]:


from scipy.stats import norm
#Visualize the distplot
sns.set(rc={'figure.figsize':(16,5)})
#to visualize the Selling_Price in the dataset
sns.distplot(df['Selling_Price'],fit=norm,kde=False,color='gray')


# In[21]:


#Visualize the distplot
sns.set(rc={'figure.figsize':(16,5)})
#to visualize the Present_Price in the dataset
sns.distplot(df['Present_Price'],fit=norm,kde=False,color='blue')


# In[22]:


#set the plot size:-
plt.figure(figsize = (15,8))

for i,z in enumerate(['Selling_Price', 'Present_Price','Driven_kms']):
    plt.subplot(2,2,i+1)
    sns.histplot(data = df, x = z,kde = True,hue = 'Transmission')
plt.show()


# BOX PLOT

# In[23]:


#set the plot size
plt.figure(figsize = (15,10))
for i,col in enumerate(['Selling_Price','Present_Price', 'Driven_kms']):
    for j,col2 in enumerate(['Transmission', 'Owner']):
        plt.subplot(3,2,i * 2 + j + 1)
        sns.boxplot(data = df, y = col2,x = col,orient = 'h')


# PAIR PLOT

# In[24]:


sns.pairplot(data = df , hue = 'Transmission')


# OUTLIERS DETECTION

# In[25]:


def outliers(col):
    per25 = df[col].quantile(0.25)
    per75 = df[col].quantile(0.75)
    IQR = per75 - per25               #Inter Quartile Range 
    UL = per75 + 1.5 * IQR            #Upper Limit
    LL = per25 - 1.5 * IQR            #Lower Limit

    return df[col]>UL


# In[26]:


df = df.drop(df[outliers('Selling_Price')].index)


# In[27]:


df = df.drop(df[outliers('Present_Price')].index)
df = df.drop(df[outliers('Driven_kms')].index)


# #AFTER DROPING OUTLIERS  IN BOX PLOT

# In[28]:


#set the plot size:-
plt.figure(figsize = (15,10))
for i,col in enumerate(['Selling_Price','Present_Price', 'Driven_kms']):
    for j,col2 in enumerate(['Transmission', 'Owner']):
        plt.subplot(3,2,i * 2 + j + 1)
        sns.boxplot(data = df, y = col2,x = col,orient = 'h')


# LABELENCODE

# In[29]:


Labelencode = LabelEncoder()


# In[30]:


cat_vars = df.select_dtypes('O').columns
for i in cat_vars:
    
    df[i] = Labelencode.fit_transform(df[i])
df.head()


# In[31]:



df.drop(columns = 'Selling_type',axis = 1,inplace=True)


# In[32]:


df.head() # data checkpoint


# In[33]:


df.shape


# SPLITTING OF DATASET

# In[34]:


Y = df['Selling_Price']
X = df.drop(columns = 'Selling_Price')


# In[35]:


from sklearn.model_selection import train_test_split
Xtrain,Xtest,ytrain,ytest = train_test_split(X,Y,test_size = 0.2, random_state = 23)


# In[36]:


from sklearn.preprocessing import MinMaxScaler
scale = MinMaxScaler()
Xtrain = scale.fit_transform(Xtrain)


# In[37]:


Xtest = scale.transform(Xtest)


# 
# MODEL SELECTION AND PREDICTION

# In[38]:


#Install the LinearRegression from sklearn:-
from sklearn.linear_model import LinearRegression
model = LinearRegression()
# fit the train data to the model
model.fit(Xtrain, ytrain)
#Prediction of the test dataset
ypred = model.predict(Xtest)

#Check the test score and train score to the LinearRegression algorithm
print(f'The Test_accuracy: {model.score(Xtest,ytest)*100:.2f}')
#Train score for the data
print(f'The Train_accuracy: {model.score(Xtrain,ytrain)*100:.2f}')


# In[39]:


#LinearRegression mean_squared_error , r2_score
from sklearn.metrics import mean_squared_error, r2_score
mse = mean_squared_error(ytest, ypred)
r2 = r2_score(ytest, ypred)

print(f"Mean Squared Error (MSE): {mse:.2f}")
print(f"R-squared (R2): {r2:.2f}")


# In[40]:


#Graphical representation:-
plt.figure(figsize=(15, 6))
plt.scatter(ytest, ypred, alpha=0.5)
plt.xlabel('Actual Selling Price')
plt.ylabel('Predicted Selling Price')
plt.title('Actual vs. Predicted Selling Prices (Linear Regression)')
plt.show()


# ploting actual selling price and predicted selling price(Linear Regression) in graphical representation. 

# In[41]:


#Install the decisiontreeregressor from sklearn
from sklearn.tree import DecisionTreeRegressor
#install the model
tree=DecisionTreeRegressor(random_state=0)
# fit the train data to the model
tree.fit(Xtrain,ytrain)
#Prediction of the test dataset
tree_pred=tree.predict(Xtest)


# In[42]:


#Check the test score and train score to the DecisionTreeRegressor algorithm
print(f'The Test_accuracy: {tree.score(Xtest,ytest)*100:.2f}')
#Train score for the data
print(f'The Train_accuracy: {tree.score(Xtrain,ytrain)*100:.2f}')


# In[43]:


#DecisionTreeRegressor mean_squared_error , r2_score
mse=mean_squared_error(ytest,tree_pred)
rmse=np.sqrt(mse)
print("Root_mean_squred_error DecisionTreeRegressor {:.4f}".format(rmse))
print("R2_score DecisionTreeRegressor {:4f}".format(r2_score(ytest,tree_pred)))
print("mean_absolute_error DecisionTreeRegressor {:4f}".format(mean_squared_error(ytest,tree_pred)))


# In[44]:


#Graphical representation:-
plt.scatter(ytest, tree_pred)
plt.xlabel("Actual Price")
plt.ylabel("Predicted Price")
plt.title(" Actual Prices vs Predicted Prices")
plt.show()


# ploting actual selling price and predicted selling price(DecisionTreeRegressor) in graphical representation. 
