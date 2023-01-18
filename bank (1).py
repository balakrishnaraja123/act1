#!/usr/bin/env python
# coding: utf-8

# In[193]:


import pandas as pd


# In[194]:


df=pd.read_excel('Bank_Personal_Loan_Modelling.xlsx')


# In[195]:


df['Personal Loan'].value_counts()


# In[196]:


df.drop(['ID','ZIP Code'],axis=1,inplace=True)


# In[197]:


#num_df = df.select_dtypes(include=['int','float'])
cat_df = df.select_dtypes(include=['category'])


# In[198]:


nume=df.select_dtypes(include=('int','float64')).columns
df[nume]=df[nume].astype('int')


# In[199]:


y=df['Personal Loan']
x=df.drop(['Personal Loan'],axis=1)


# In[200]:


from sklearn.model_selection import train_test_split


# In[201]:


x_train,x_test,y_train,y_test,=train_test_split(x,y,test_size=0.3,random_state=101)


# In[202]:


print(x_train.shape)
print(y_train.shape)


# In[203]:


nume=['Age','Income','Experience','CCAvg','Mortgage']


# In[204]:


from sklearn.preprocessing import StandardScaler
scaler=StandardScaler()
scaling=scaler.fit(x_train[nume])


# In[205]:


sca_tr=scaler.transform(x_train[nume])
sca_te=scaler.transform(x_test[nume])


# In[206]:


scal_df1=pd.DataFrame(sca_tr)
scal_df2=pd.DataFrame(sca_te)


# In[207]:


from sklearn.preprocessing import OneHotEncoder


# In[208]:


xtr_oh=OneHotEncoder()
xtr_oh.fit(x_train[cat_df])


# In[209]:


xtr_trs=xtr_oh.transform(x_train[cat_df]).toarray()
xte_trs=xtr_oh.transform(x_test[cat_df]).toarray()


# In[210]:


xtr_oh_df=pd.DataFrame(xtr_trs)
xte_oh_df=pd.DataFrame(xte_trs)


# In[211]:


xtr_con=pd.concat([scal_df1,xtr_oh_df],axis=1)
xte_con=pd.concat([scal_df2,xte_oh_df],axis=1)


# In[212]:


def evaluate_model(act, pred):
    from sklearn.metrics import confusion_matrix, accuracy_score, recall_score, precision_score,f1_score
    print("Confusion Matrix \n", confusion_matrix(act, pred))
    print("Accurcay : ", accuracy_score(act, pred))
    print("Recall   : ", recall_score(act, pred))
    print("Precision: ", precision_score(act, pred))  
    print('F1 Score:\n',f1_score(act,pred))


# In[213]:


from sklearn.linear_model import LogisticRegression


# In[214]:


log=LogisticRegression()


# In[215]:


log.fit(xtr_con,y_train)


# In[216]:


predic_xtr=log.predict(xtr_con)
predic_xte=log.pr`edict(xte_con)


# In[224]:


pd.DataFrame.from_dict(predic_xte)


# In[223]:


predic_xte.to_csv


# In[217]:


evaluate_model(y_train,predic_xtr)
evaluate_model(y_test,predic_xte)


# In[218]:


import pickle


# In[219]:


print('saving model as pkl file.......')
pickle.dump(log, open('model.pkl','wb'))


# In[220]:


model = pickle.load(open('model.pkl','rb'))


# In[ ]:




