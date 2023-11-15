#!/usr/bin/env python
# coding: utf-8

# ###DURGAVENI RAVIPATI | DATA SCIENCE INTERN | OASIS INFOBYTE | #TASK 2

# UNEMPLOYMENT ANALYSIS WITH PYTHON

# In[1]:


# IMPORT NECESSARY LIBRARIES
#import numpy libreary as np
import numpy as np
#import pandas as pd
import pandas as pd
#import matplotlib.pyplot as plt
import matplotlib.pyplot as plt
#import seaborn as sns
import seaborn as sns
#feature warnings
import warnings
warnings.filterwarnings("ignore")
import plotly.express as px
import datetime as dt
import calendar
from sklearn import metrics


# In[2]:


#Load car dataset:-
df = pd.read_csv('Unemployment_Rate_upto_11_2020.csv')


# In[3]:


# rows of dataset:- 
print("Rows from start are: ")
print(df.head(10))
print("\n")
print("Rows from bottom: ")
print(df.tail(10))


# In[4]:


# shape of the dataset:-
df.shape


# In[5]:


#information of the dataset:-
df.info()


# In[6]:


#data types by each column
df.dtypes


# In[7]:


#describe the dataset:-
df.describe(include='object')


# HANDLING MISSING DATA

# In[8]:


df.isnull().sum()


# In[9]:


#check duplicate values:-
df.duplicated().sum()


# There is no null values present in the dataset.

# In[10]:


#correlation:-
corr =df.corr()
corr


# In[11]:


#plot a heatmap for the correlation matrix
plt.style.use('seaborn-whitegrid')
plt.figure(figsize=(12, 10))
sns.heatmap(df.corr())
plt.show()


# In[12]:


df[' Frequency'] = df[' Frequency'].map({" Monthly":"/Month",
                                             
                                            "Monthly":"/Month"})


# In[13]:


df[' Frequency'].value_counts()


# In[14]:


df.columns = ['States', 'Date', 'Frequency', 'Estimated Unemployment Rate', 'Estimated Employed',
              'Estimated Labour Participation Rate', 'Region', 'longitude', 'latitude']

# Converting 'Date' column to datetime format
df['Date'] = pd.to_datetime(df['Date'], dayfirst=True)

# Converting 'Frequency' and 'Region' columns to categorical data type
df['Frequency'] = df['Frequency'].astype('category')
df['Region'] = df['Region'].astype('category')

# Extracting month from 'Date' and creating a 'Month' column
df['Month'] = df['Date'].dt.month

# Converting 'Month' to integer format
df['Month_int'] = df['Month'].apply(lambda x: int(x))

# Mapping integer month values to abbreviated month names
df['Month_name'] = df['Month_int'].apply(lambda x: calendar.month_abbr[x])
df.drop(columns= 'Month',inplace = True)


# In[15]:


#Assuming you have a DataFrame named 'df' with 'Region','Estimated Labour Participation Rate' columns
sum_employees_by_region = df.groupby('Region')['Estimated Labour Participation Rate'].sum().reset_index()
#'sum _employees_by_region' contains the sum of labour participation for each region
print(sum_employees_by_region)


# North region has highest estimated labour participation rate during covid 19 pandamic. 

# In[16]:


#Assuming you have a DataFrame named 'df' with 'Region','Estimated Employed' columns
sum_employees_by_region = df.groupby('Region')['Estimated Employed'].sum().reset_index()
#'sum _employees_by_region' contains the sum of employess for each region
print(sum_employees_by_region)


# North region has highest estimated employees during covid 19 pandamic.

# In[17]:


#Assuming you have a DataFrame named 'df' with 'Region','Estimated Unemployment Rate' columns
sum_employees_by_region = df.groupby('Region')['Estimated Unemployment Rate'].sum().reset_index()
#'sum _employees_by_region' contains the sum of unemployess for each region
print(sum_employees_by_region)


# North region has highest estimated unemployment rate during covid 19 pandamic.

# EXPLORATORY DATA ANALYSIS

# In[18]:


##Visualize the pairplot:-
sns.pairplot(df)
plt.show()


# In[19]:


#Visualize the histogram :-
# Histogram of Estimated Unemployment Rate,Estimated Employed, Estimated Labour Participation Rate by Region:-
plt.figure(figsize=(20, 10))
for i,z in enumerate(['Estimated Unemployment Rate','Estimated Employed', 'Estimated Labour Participation Rate']):
    plt.subplot(2,2,i+1)
    sns.histplot(x=z, hue='Region', data=df, kde=True, palette="Set1")
    
    plt.ylabel("Count")
plt.title("Histogram of Estimated Employment Rate by Region")
plt.show()


# BOX PLOT

# In[21]:


#Visualize the boxplot 
#chart showing outliers of uneployement rate in each region and state

plt.figure(figsize = (15,6))
sns.boxplot(x='States', y='Estimated Unemployment Rate', data=df)
plt.title('Box Plot of Estimated Unemployment Rate by states')
plt.xticks(rotation=45)
plt.show()


# In[22]:


#Visualize the boxplot :-
#chart showing outliers of  labour participation rate in each region and state
plt.figure(figsize = (15,6))
sns.boxplot(x='States', y='Estimated Labour Participation Rate', data=df)
plt.title('Box Plot of Estimated Labour Participation Rate by Region')
plt.xticks(rotation=45)
plt.show()


# In[23]:


#Visualize the lineplott :-
#chart showing Unemployment according to Date:
fig = plt.figure(figsize = (8, 6))
sns.lineplot(y='Estimated Unemployment Rate', x='Date', data=df)
plt.title('Unemployment according to Date')
plt.xlabel('Date')
plt.xticks(rotation=90)
plt.ylabel('Unemployment Rate')
plt.show()


# In[24]:


#Visualize the lineplott
#Now let’s see the unemployment rate according to different regions of India:
fig = plt.figure(figsize = (8, 6))
sns.lineplot(y='Estimated Unemployment Rate', x='States', data=df)
plt.title('Unemployment according to States')
plt.xlabel('States')
plt.xticks(rotation=90)
plt.ylabel('Unemployment Rate')
plt.show()


# In[25]:


#Visualize the lineplott 
#Now let’s see the labour participation rate according to different regions of India:
fig = plt.figure(figsize = (8, 6))
sns.lineplot(y='Estimated Labour Participation Rate', x='States', data=df)
plt.title('Labour Participation according to States')
plt.xlabel('States')
plt.xticks(rotation=90)
plt.ylabel('Labour Participation')
plt.show()


# HISTPLOT

# In[26]:


#Visualize the histplot :-
#Now let’s see the unemployment rate according to different regions of India:
plt.figure(figsize=(12, 11))
plt.title('Indian Unemployment')
sns.histplot(x='Estimated Unemployment Rate', hue='Region', data=df)
plt.show()


# In[27]:


#Now let’s see the labour participation rate according to different regions of India:
plt.figure(figsize=(12, 11))
plt.title('Indian Labour Participation')
sns.histplot(x='Estimated Labour Participation Rate', hue='Region', data=df)
plt.show()


# In[28]:


#Visualize the point plot
#Now let’s see the labour participation rate according to different regions of India:
sns.pointplot(y='States', x='Estimated Labour Participation Rate', data=df)


# In[29]:


#Visualize the point plot with out joining the plots.
sns.pointplot(y='States', x='Estimated Labour Participation Rate', data=df, join=False)


# In[30]:


#Visualize the swarm plot
#Now let’s see the labour participation rate according to different regions of India:
sns.swarmplot(y='States', x='Estimated Labour Participation Rate', data=df)


# PIE CHART

# In[31]:


#Visualize the pie plot
#Now let’s see the  unemployment rate according to different regions of India:
plt.figure(figsize =  (15,6))
df_grouped = df.groupby('States')['Estimated Unemployment Rate'].sum()
df_grouped.plot(kind='pie', autopct='%1.1f%%')
plt.title('Distribution of Estimated Unemployment Rate by Region')
plt.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.
plt.show()


# In[32]:


#Visualize the pie plot :-
#Now let’s see the labour participation rate according to different regions of India:
plt.figure(figsize =  (15,6))
df_grouped = df.groupby('States')['Estimated Labour Participation Rate'].sum()
df_grouped.plot(kind='pie', autopct='%1.1f%%')
plt.title('Distribution of Estimated Labour Participation Rate by Region')
plt.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.
plt.show()


# In[33]:


#Visualize the pie plot
#Now let’s see the employed rate according to different regions of India:
plt.figure(figsize =  (15,6))
df_grouped = df.groupby('States')['Estimated Employed'].sum()
df_grouped.plot(kind='pie', autopct='%3.1f%%')
plt.title('Distribution of Estimated Employment by Region')
plt.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.
plt.show()


# In[34]:


#Visualize the scatter plot
fig = px.scatter_geo(df,'longitude', 'latitude', color="Region",
                     hover_name="States", size="Estimated Unemployment Rate",
                     animation_frame="Month_name",scope='asia',template='seaborn',title='Impack of lockdown on Employement across regions')

fig.layout.updatemenus[0].buttons[0].args[1]["frame"]["duration"] = 3000

fig.update_geos(lataxis_range=[5,35], lonaxis_range=[65, 100],oceancolor="#3399FF",
    showocean=True)

fig.show()


# In[35]:


# Sunburst chart showing Labour Participation rate in each region and state

df2 = df[['States', 'Region', 'Estimated Unemployment Rate', 'Estimated Employed', 'Estimated Labour Participation Rate']]
unemplo = df2.groupby(['Region', 'States'])['Estimated Labour Participation Rate'].mean().reset_index()
fig = px.sunburst(unemplo, path=['Region', 'States'], values='Estimated Labour Participation Rate',
                  color_continuous_scale='Plasma', title='Labour Participation rate in each region and state',
                  height=650, template='ggplot2')
fig.show()


# In[36]:


# Sunburst chart showing employment rate in each region and state

df2 = df[['States', 'Region', 'Estimated Unemployment Rate', 'Estimated Employed', 'Estimated Labour Participation Rate']]
unemplo = df2.groupby(['Region', 'States'])['Estimated Employed'].mean().reset_index()
fig = px.sunburst(unemplo, path=['Region', 'States'], values='Estimated Employed',
                  color_continuous_scale='Plasma', title='Employment in each region and state',
                  height=650, template='ggplot2')
fig.show()


# In[37]:


# Sunburst chart showing unemployment rate in each region and state
un = df[['States', 'Region', 'Estimated Unemployment Rate']]
figure = px.sunburst(un, path=[ 'Region','States'],
                    values='Estimated Unemployment Rate',
                    width=700, height=700, color_continuous_scale='RdY1Gn',
                    title='uneployement rate in india')
figure.show()


# In[38]:


# Distribution of Estimated Labour Participation Rate by Region

df3 = df[['Region', 'Estimated Labour Participation Rate']]
df3_grouped = df3.groupby('Region')['Estimated Labour Participation Rate'].sum()
df_grouped.plot(kind='pie', autopct='%1.1f%%')
plt.title('Distribution of Estimated Labour Participation Rate by Region')
plt.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.
plt.show()


# In[39]:


# Distribution of Estimated Unemployment Rate by Region

df = df[['Region', 'Estimated Unemployment Rate']]
df2_grouped = df2.groupby('Region')['Estimated Unemployment Rate'].sum()
df2_grouped.plot(kind='pie', autopct='%1.1f%%')
plt.title('Distribution of Estimated Unemployment Rate by Region')
plt.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.
plt.show()


# In[40]:


# BoxPlot of Estimated Labour Participation Rate by Region

import seaborn as sns

#df3 = df[['Region', 'Estimated Labour Participation Rate']]
sns.boxplot(x='Region', y='Estimated Labour Participation Rate', data=df3)
plt.xlabel('Region')
plt.ylabel('Estimated Labour Participation Rate')
plt.title('Box Plot of Estimated Labour Participation Rate by Region')
plt.xticks(rotation=45)
plt.show()


# In[41]:


# BoxPlot of Estimated Estimated Employed by Region

import seaborn as sns

df = df[['Region', 'Estimated Unemployment Rate']]
sns.boxplot(x='Region', y='Estimated Unemployment Rate', data=df)
plt.xlabel('Region')
plt.ylabel('Estimated Unemployment Rate')
plt.title('Box Plot of Estimated Unemployment Rate by Region')
plt.xticks(rotation=45)
plt.show()

