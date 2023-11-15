#!/usr/bin/env python
# coding: utf-8

# ###DURGAVENI RAVIPATI | DATA SCIENCE INTERN | OASIS INFOBYTE | #TASK 1

# IRIS FLOWER CLASSIFICATION

# In[1]:


# IMPORT NECESSARY LIBRARIES
#import pandas as pd
import pandas as pd
#import numpy libreary as np
import numpy as np
#import matplotlib.pyplot as plt
import matplotlib.pyplot as plt
#import seaborn as sns
import seaborn as sns
from pandas.plotting import scatter_matrix


# In[2]:


#Load car dataset:-
df =pd.read_csv("Iris.csv")


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


# In[9]:


#Drop the column:-
df = df.drop(columns = 'Id')


# In[10]:


#Changing the column names:-
colnames = ['Sepal_L', 'Sepal_W', 'Petal_L', 'Petal_W', 'Species']


# In[11]:


df.columns = colnames


# In[12]:


df["Species"].describe()


# In[13]:


print(df.groupby("Species").size())


# In[14]:


df.isnull().sum()


# In[15]:


#correlation:-
df.corr(method = "pearson")


# In[16]:


#Skewness:-
df.skew(numeric_only = True)


# EXPLORATORY DATA ANALYSIS

# HISTPLOT

# In[17]:


#Visualize the histplot:-
df.hist()
plt.show()


# DENSITY PLOT

# In[18]:


#Visualize the densityplot :-
df.plot(kind = "density", subplots = True, layout = (2,2), sharex = False, sharey = False)
plt.show()


# SCATTER PLOT

# In[19]:


#Visualize the scatter plot :-
scatter_matrix(df)
plt.show()


# PAIR PLOT

# In[20]:


#Visualize the pairplot:-
sns.pairplot(df,hue='Species')


# DATA PREPARATION

# In[21]:


#LABELENCODE:-
from sklearn.preprocessing import LabelEncoder

encoder = LabelEncoder()
encoder.fit(df['Species'])
species_dict = dict(enumerate(encoder.classes_))
species_dict


# In[22]:


encoded_species = encoder.fit_transform(df['Species'])
encoded_species[:4]


# In[23]:


df1 = df.copy() # data checkpoint
df1['Species'] = encoded_species

df1.head()


# In[24]:


X = df1.drop(['Species'], axis=1)
y = df1['Species']


# SPLITTING OF DATASET

# In[25]:


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.2, random_state=36)


# In[26]:


#data transformation/scaling(standardization)
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.fit_transform(X_test)


# 
# MODEL SELECTION AND PREDICTION

# In[27]:


from sklearn.metrics import confusion_matrix ,accuracy_score
def evaluateModel(y_test, y_pred):
    print("Accuracy : {}".format(accuracy_score(y_test, y_pred)))
    print("\n")


# In[28]:


#Install the LogisticRegression from sklearn:-
from sklearn.linear_model import LogisticRegression
logreg = LogisticRegression()
# fit the train data to the model
logreg.fit(X_train,y_train)

print('Logistic Regression')
print('Train Score: ')
y_train_pred = logreg.predict(X_train)
evaluateModel(y_train,y_train_pred)
print('Test Score:')
y_test_pred = logreg.predict(X_test)
evaluateModel(y_test,y_test_pred)


# In[29]:


from sklearn.metrics import confusion_matrix
confusion_matrix1=confusion_matrix(y_train,y_train_pred)
confusion_matrix2=confusion_matrix(y_test,y_test_pred)
from sklearn.metrics import ConfusionMatrixDisplay
cm = ConfusionMatrixDisplay(confusion_matrix = confusion_matrix1)
cm2 = ConfusionMatrixDisplay(confusion_matrix = confusion_matrix2)

cm.plot()
cm2.plot()
plt.title('Confusion Matrix')
plt.show()


# In[30]:


#Install the DecisionTreeClassifier from sklearn:-
from sklearn.tree import DecisionTreeClassifier
dt = DecisionTreeClassifier()
dt.fit(X_train, y_train)

print('Decision Tree')
print('Train Score: ')
y_train_pred = dt.predict(X_train)
evaluateModel(y_train,y_train_pred)
print('Test Score : ')
y_test_pred = dt.predict(X_test)
evaluateModel(y_test,y_test_pred)


# In[31]:


from sklearn.metrics import confusion_matrix
confusion_matrix1=confusion_matrix(y_train,y_train_pred)
confusion_matrix2=confusion_matrix(y_test,y_test_pred)
from sklearn.metrics import ConfusionMatrixDisplay
cm = ConfusionMatrixDisplay(confusion_matrix = confusion_matrix1)
cm2 = ConfusionMatrixDisplay(confusion_matrix = confusion_matrix2)

cm.plot()
cm2.plot()
plt.title('Confusion Matrix')
plt.show()


# In[32]:


#Install the RandomForestClassifier from sklearn:-
from sklearn.ensemble import RandomForestClassifier
rf = RandomForestClassifier()
rf.fit(X_train, y_train)

print('Random Forest')
print('Train Score:')
y_train_pred = rf.predict(X_train)
evaluateModel(y_train,y_train_pred)
print('Test Score : ')
y_test_pred = rf.predict(X_test)
evaluateModel(y_test,y_test_pred)


# In[33]:


from sklearn.metrics import confusion_matrix
confusion_matrix1=confusion_matrix(y_train,y_train_pred)
confusion_matrix2=confusion_matrix(y_test,y_test_pred)
from sklearn.metrics import ConfusionMatrixDisplay
cm = ConfusionMatrixDisplay(confusion_matrix = confusion_matrix1)
cm2 = ConfusionMatrixDisplay(confusion_matrix = confusion_matrix2)

cm.plot()
cm2.plot()
plt.title('Confusion Matrix')
plt.show()

