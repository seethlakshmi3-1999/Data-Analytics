#!/usr/bin/env python
# coding: utf-8

# In[4]:


True /10


# In[5]:


not 2>1


# In[17]:


app=3000
ex=5000


# In[14]:


na=90


# In[15]:


grape=80


# In[18]:


total=ex-app
total


# In[22]:


print("I\'m seetha",
"how you doing?",sep="...",end="(<_>)")


# In[37]:


g=["hello"
,"worls"]
print(g)


# In[42]:


h=g
h


# In[41]:


h.remove("hello")


# In[45]:


dic={"ram":69,
    "sachi":78,
    "loki":90}
print(dic)


# In[50]:


for k in dic:
    print("name:",k,",","age:",dic[k])


# In[ ]:


d=6
while d:
    print("Hi",d)
    d+=1


# In[5]:


import matplotlib.pyplot as plt
import numpy as np
n=100
x=np.random.rand(n)
y=np.random.rand(n)
colors=np.random.rand(n)
area=(30 * np.random.rand(n))**2
plt.figure(figsize=(15,8))
plt.scatter(x,y,s=area,c=colors,alpha=.5)
plt.show()


# In[8]:


get_ipython().run_line_magic('time', '')
for i in range(5):
    print(i)


# In[9]:


get_ipython().system('pip install seaborn')


# In[11]:


get_ipython().system('pip install scikit-learn')


# In[23]:


import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns


# In[24]:


marks=pd.read_csv("downloads/marksheet.csv",encoding='utf-8',index_col=0)
marks.head()


# In[25]:


marks.shape


# In[26]:


marks.drop(["Total"],inplace=True,axis=1)


# In[27]:


marks.sample(10)


# In[28]:


marks["RollNumber"].unique()


# In[29]:


marks.RollNumber.value_counts()


# In[30]:


from sklearn.preprocessing import LabelEncoder
label_encoder=LabelEncoder()
marks["RollNumber"]=label_encoder.fit_transform(marks["RollNumber"])


# In[31]:


marks.head(5)


# In[32]:


dummy=pd.get_dummies(marks)
dummy.head()


# In[ ]:





# In[33]:


dummy.shape


# In[39]:


X=dummy.drop('mark1',axis=1)
y=dummy['mark1']


# In[40]:


from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2)


# In[41]:


X_train.shape,X_test.shape


# In[42]:


y_train.shape,y_test.shape


# In[44]:


from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
linear_regression =LinearRegression()
linear_regression.fit(X_train, y_train)


# In[45]:


linear_regression.score(X_train,y_train)


# In[46]:


y_pred=linear_regression.predict(X_test)
r2_score(y_test,y_pred)


# In[47]:


plt.figure(figsize=(11,6))
plt.scatter(y_test,y_pred)
plt.xlabel('Actual Value')
plt.ylabel('predicted value')
plt.show


# In[48]:


get_ipython().system('pip install ipywidgets')


# In[49]:


get_ipython().system('jupyter nbextension enable --py widgetsnbextension --sys-prefix')


# In[50]:


from ipywidgets import interact,interactive,fixed,interact_manual
import ipywidgets as widgets


# In[54]:


w=widgets.IntSlider(value=10,
                   min=-5,
                   max=20,
                   step=1,
                   description='Range SLider:',
                   continuous_update=False,
                   orientation='horizontal')


# In[55]:


w


# In[56]:


w.value


# In[57]:


type(w)


# In[60]:


r=widgets.IntRangeSlider(value=[10,1000],
                   min=0,
                   max=1000,
                   step=1,
                   description='Price Range',
                   orientation='horizontal')


# In[61]:


r


# In[66]:


p=widgets.IntProgress(value=70,
                   min=0,
                   max=100,
                   step=1,
                   description='Loading',
                   bar_style='Success',
                   orientation='horizontal')


# In[67]:


p


# In[68]:


import time
for i in range(0,110,10):
    p.value=i
    time.sleep(1)


# In[84]:


button=widgets.Button(description='Click me',
                      button_style='success',
                      tooltip='god things will happen',
                      icon='check'
                     )


# In[85]:


def button_click(button):
    print('Have a good day')
    print(button.description)
    


# In[86]:


button.on_click(button_click)


# In[87]:


button


# In[99]:


play=widgets.Play(value=50,
                 min=0,
                 max=100,
                 step=1,
                 description="press play")
slider= widgets.IntSlider()
widgets.jslink((play,'value'),(slider,'value'))
widgets.HBox([play,slider])


# In[100]:


def f(x):
    return x


# In[101]:


interact(f,x=10);


# In[102]:


interact(f,x=True)


# In[104]:


interact(f,x="HEllo");
    


# In[114]:


import os
from IPython.display import Image
fdir='Downloads/'
@interact
def show_image(file = os.listdir(fdir)):
    display(Image(fdir +file))


# In[ ]:




