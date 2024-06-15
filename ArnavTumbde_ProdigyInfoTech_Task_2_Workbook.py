#!/usr/bin/env python
# coding: utf-8

# ## Arnav Tumbde
# ## Domain : Data Science
# ## Task : 2

# ### Loading the Dataset

# In[3]:


import pandas as pd
titanic = pd.read_csv("D:\\Semester - IV\\Prodigy Internship\\Task 2\\archive\\Titanic-Dataset.csv")


# ### Data Cleaning

# #### Check for Missing Values

# In[4]:


titanic.isnull().sum()


# #### Handling Missing Values

# In[9]:


titanic['Age'].fillna(titanic['Age'].median(), inplace=True)
titanic.dropna(subset=['Embarked'], inplace=True)
titanic


# #### Data type conversions if necessary

# In[10]:


titanic['Fare'] = titanic['Fare'].astype(float)
titanic


# #### Remove Duplicates

# In[11]:


titanic.drop_duplicates(inplace=True)
titanic


# #### Data Characteristics Operations

# In[23]:


male_ind = len(titanic[titanic['Sex'] == 'male'])
print("No of Males in Titanic:",male_ind)


# In[25]:


female_ind = len(titanic[titanic['Sex'] == 'female'])
print("No of Females in Titanic:",female_ind)


# In[26]:


titanic.shape


# In[32]:


alive = len(titanic[titanic['Survived'] == 1])
dead = len(titanic[titanic['Survived'] == 0])
print("Alive => ", alive)
print("Dead => ", dead)


# ### EDA Performing

# In[8]:


titanic.describe()


# #### Visualization Using Seaborn and MatplotLib

# In[15]:


import seaborn as sns
import matplotlib.pyplot as plt


# In[33]:


fig = plt.figure()
ax = fig.add_axes([0,0,1,1])
gender = ['Male','Female']
index = [577,314]
ax.bar(gender,index)
plt.xlabel("Gender")
plt.ylabel("No of people onboarding ship")
plt.show()


# In[35]:


plt.figure(1)
titanic.loc[titanic['Survived'] == 1, 'Pclass'].value_counts().sort_index().plot.bar()
plt.title('Bar graph of people accrding to ticket class in which people survived')


plt.figure(2)
titanic.loc[titanic['Survived'] == 0, 'Pclass'].value_counts().sort_index().plot.bar()
plt.title('Bar graph of people accrding to ticket class in which people couldn\'t survive')


# In[16]:


sns.histplot(titanic['Age'], bins=30, kde=True)
plt.title('Age Distribution')
plt.show()


# In[17]:


sns.barplot(x='Pclass', y='Survived', data=titanic)
plt.title('Survival Rate by Class')
plt.show()


# In[22]:


# Violin plot for Age distribution by class and survival status
plt.figure(figsize=(10, 6))
sns.violinplot(x='Pclass', y='Age', hue='Survived', data=titanic, split=True, palette='coolwarm')
plt.title('Age Distribution by Class and Survival Status')
plt.show()


# In[36]:


fig = plt.figure()
ax = fig.add_axes([0,0,1,1])
ax.axis('equal')
l = ['C = Cherbourg', 'Q = Queenstown', 'S = Southampton']
s = [0.553571,0.389610,0.336957]
ax.pie(s, labels = l,autopct='%5.3f%%')
plt.show()


# #### Exploring Relationships and Identifying Patterns

# In[20]:


survival_rate_by_sex = titanic.groupby('Sex')['Survived'].mean()
survival_rate_by_class = titanic.groupby('Pclass')['Survived'].mean()
survival_rate_by_age = titanic.groupby(pd.cut(titanic['Age'], bins=[0, 12, 18, 50, 80]))['Survived'].mean()
print(survival_rate_by_sex, survival_rate_by_class, survival_rate_by_age)


# ### Conclusion : Thus performed Data Cleaning as well as EDA on Titanic-Dataset exploring Relationship and identifying patterns along with visualizing the dataset in unique ways 

# In[ ]:




