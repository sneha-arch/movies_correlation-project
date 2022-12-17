#!/usr/bin/env python
# coding: utf-8

# In[31]:


import pandas as pd
import numpy as np
import seaborn as sns

import matplotlib.pyplot as plt
import matplotlib.mlab as mlab
import matplotlib
plt.style.use('ggplot')
from matplotlib.pyplot import figure

get_ipython().run_line_magic('matplotlib', 'inline')
matplotlib.rcParams['figure.figsize'] = (12,8)

pd.options.mode.chained_assignment = None


# In[43]:


# Now we need to read in the data
df = pd.read_csv(r'C:\Users\abhis\OneDrive\Documents\movies.csv')


# In[50]:


# Now let's take a look at the data

df.head()


# In[5]:


# We need to see if we have any missing data
# Let's loop through the data and see if there is anything missing

for col in df.columns:
    pct_missing = np.mean(df[col].isnull())
    print('{} - {}%'.format(col, round(pct_missing*100)))


# In[6]:


# Data Types for our columns

print(df.dtypes)


# In[15]:



# Are there any Outliers?

df.boxplot(column=['gross'])


# In[35]:


#create correct year column
df['yearcorrect'] = df['released'].astype(str).str[:4]
df


# In[41]:


df=df.sort_values(by=['gross'],inplace=False,ascending=False)
pd.set_option('display.max_rows',None) #to display the whole dataset


# In[38]:


df.drop_duplicates()


# In[44]:


#scatter plot with budget vs gross
plt.scatter(x=df['budget'],y=df['gross'])
plt.title('budget vs gross earnings')
plt.xlabel('gross earnings')
plt.ylabel('budget')
plt.show()


# In[43]:


df.head()


# In[48]:


#plot gross vs budget using seaburn to see the correlation
sns.regplot(x="gross", y="budget", data=df, scatter_kws={"color":"red"}, line_kws={"color":"blue"})


# In[21]:


df.corr(method='pearson')


# In[ ]:





# In[ ]:





# In[25]:


# there is a high correlation between gross and budget.
correlation_matrix = df.corr(method='pearson')
sns.heatmap(correlation_matrix, annot= True)
plt.title('correlation matrix')
plt.xlabel('movie features')
plt.ylabel('movie features')
plt.show()


# In[40]:


# changing string columns into numeric
df_numerized= df
for col_name in df_numerized.columns:
    if (df_numerized[col_name].dtype== 'object'):
        df_numerized[col_name]= df_numerized[col_name].astype('category')
        df_numerized[col_name]= df_numerized[col_name].cat.codes
        df_numerized


# In[39]:


pd.set_option('display.max_rows',None) #to display whole dataset
df


# In[51]:


df.head()


# In[45]:


correlation_matrix = df_numerized.corr(method='pearson')
sns.heatmap(correlation_matrix, annot= True)
plt.title('correlation matrix')
plt.xlabel('movie features')
plt.ylabel('movie features')
plt.show()


# In[46]:


df_numerized.corr()


# In[47]:


correlation_mat=df_numerized.corr()
corr_pairs=correlation_mat.unstack()
corr_pairs


# In[48]:


sorted_pairs=corr_pairs.sort_values()
sorted_pairs


# In[49]:


high_corr= sorted_pairs[(sorted_pairs)>0.5]
high_corr


# In[ ]:


#votes and budget have highest correlation with gross earnings.
#company has low correlation. i was wrong.

