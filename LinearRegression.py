#!/usr/bin/env python
# coding: utf-8

# # Packages Import and Cleaning and Normalization of Data

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# In[2]:


project_df = pd.read_csv("sgemm_product_dataset\sgemm_product.csv")


# In[3]:


project_df['Run_Avg'] = project_df.iloc[:,14:18].mean(axis=1)


# In[4]:


project_df=project_df.drop(columns=['Run1 (ms)','Run2 (ms)','Run3 (ms)','Run4 (ms)'])


# In[5]:


project_df=project_df.dropna()
#project_df


# In[6]:


project_df['Run_Avg'] = np.log(project_df['Run_Avg'])


# In[7]:


normalized_df = (project_df - project_df.mean())/project_df.std()
normalized_df


# # Train-test data split and Gradient descent calculation Functions

# In[8]:


def train_test(x):
    #x is the dataset you want to split and use for regression
    num_of_rows=int(len(x)*0.65)
    shuffle_project_df = x.sample(frac=1)
    training_set = shuffle_project_df[:num_of_rows]
    test_set = shuffle_project_df[num_of_rows:]
    return training_set,test_set


# In[9]:


def gradient_descent(X,Y,beta,bias,lr,threshold):
    m=len(X)
    
    array_MSE=[]
    array_iteration=[]

    Y_pred = np.dot(X,beta) + bias
    count=0
    
    for i in range(1000):
        count+=1
        D_beta = (1/m)*lr*(X.T.dot(Y_pred - Y)) #derivative with respect to beta
        D_bias = (1/m)*lr*(np.sum(Y_pred - Y)) #derivative with respect to bias
        
        Y_pred = np.dot(X,beta) + bias
        beta = beta - D_beta
        bias = bias - D_bias
        MSE = (1/m) * np.sum(np.square(Y_pred-Y))
        array_MSE.append(MSE)
        array_iteration.append(i)
        
        if D_beta.mean() < threshold:
            count-=1
              
    return MSE, count, beta, bias, array_MSE, array_iteration


# # Part 1: All 14 Features Selected with Multiple Learning Rates

# In[10]:


normalized_df


# In[11]:


training_set, test_set = train_test(normalized_df)


# In[12]:


X_train = training_set.iloc[:,:-1]
X_test = test_set.iloc[:,:-1]
Y_train = training_set['Run_Avg']
Y_test = test_set['Run_Avg']


# ## Train Data

# In[13]:


learning_rate = [0.01,0.03,0.001,0.0001]
new_MSE=[]
for lr in learning_rate:
    beta = np.c_[np.zeros(X_train.shape[1])]
    beta.shape=np.squeeze(beta).shape

    bias = np.c_[np.zeros(1)]
    bias.shape=np.squeeze(bias).shape
    
    MSE,count, beta, bias, array_MSE, array_iteration = gradient_descent(X_train,Y_train,beta,bias,lr,0.00000001)
    print(MSE)
    new_MSE.append(MSE)
    plt.plot(array_iteration,array_MSE,label=lr)
    plt.xlabel('Iterations')
    plt.ylabel('Mean Squared Error')
    plt.legend()


# In[ ]:





# ## Test Data

# In[14]:


learning_rate = [0.01,0.03,0.001,0.0001]
new_MSE2=[]
for lr in learning_rate:
    MSE,count,beta, bias, array_MSE, array_iteration = gradient_descent(X_test,Y_test,beta,bias,lr,0.00000001)
    print(MSE)
    new_MSE2.append(MSE)
    plt.plot(array_iteration,array_MSE,label=lr)
    plt.xlabel('Iterations')
    plt.ylabel('Mean Squared Error')
    plt.legend()


# In[15]:


plt.plot(np.log10(learning_rate),new_MSE,label='train')
plt.xlabel('Learning Rate in power of 10')
plt.ylabel('Mean Squared Error')
plt.plot(np.log10(learning_rate),new_MSE2,label='test')
plt.xlabel('Learning Rate in power of 10')
plt.ylabel('Mean Squared Error')
plt.legend()


# # Part 2 All Features Selected with Multiple Threshold Values

# ## Train Data

# In[16]:


array_count=[]
new_MSE=[]
learning_rate = [0.01]
threshold = [0.0001,0.00001,0.000001,0.0000001,0.00000001]
for lr in learning_rate:
    for th in threshold:
        beta = np.c_[np.zeros(X_train.shape[1])]
        beta.shape=np.squeeze(beta).shape

        bias = np.c_[np.zeros(1)]
        bias.shape=np.squeeze(bias).shape
        
        MSE,count, beta, bias, array_MSE, array_iteration = gradient_descent(X_train,Y_train,beta,bias,lr,th)
        array_count.append(count)
        new_MSE.append(array_MSE[count+1])
        print('No. of Iterations required to reach Convergence with threshold ',th,' is:', count)
        plt.plot(array_iteration[:count+1],array_MSE[:count+1],label=array_MSE[count+1])
        plt.xlabel('Iterations')
        plt.ylabel('Mean Squared Error')
        plt.legend()


# In[17]:


plt.plot(np.log10(threshold),array_count)
plt.xlabel('Threshold in power of 10')
plt.ylabel('Iterations')


# In[ ]:





# ## Test Data

# In[18]:


array_count=[]
new_MSE2=[]
learning_rate = [0.01]
threshold = [0.0001,0.00001,0.000001,0.0000001,0.00000001]
for lr in learning_rate:
    for th in threshold:
        MSE, count, beta, bias, array_MSE, array_iteration = gradient_descent(X_test,Y_test,beta,bias,lr,th)
        array_count.append(count)
        new_MSE2.append(array_MSE[count+1])
        print(MSE,th)


# In[19]:


plt.plot(np.log10(threshold),new_MSE,label='train')
plt.plot(np.log10(threshold),new_MSE2,label='test')
plt.xlabel('Threshold in power of 10')
plt.ylabel('Mean Squared Error')
plt.legend()


# # Part 3 Random 8 Features Selected

# In[20]:


updated_df = normalized_df.iloc[:,:14].sample(8,axis=1)
updated_df['Run_Avg'] = normalized_df['Run_Avg']


# In[21]:


training_set, test_set = train_test(updated_df)


# In[22]:


X_train = training_set.iloc[:,:-1]
X_test = test_set.iloc[:,:-1]
Y_train = training_set['Run_Avg']
Y_test = test_set['Run_Avg']


# ## Train Data

# In[23]:


learning_rate = [0.01]
for lr in learning_rate:
    beta = np.c_[np.zeros(X_train.shape[1])]
    beta.shape=np.squeeze(beta).shape

    bias = np.c_[np.zeros(1)]
    bias.shape=np.squeeze(bias).shape
    
    MSE, count, beta, bias, array_MSE, array_iteration = gradient_descent(X_train,Y_train,beta,bias,lr,0.00000001)
    print(lr, MSE)
    plt.plot(array_iteration,array_MSE,label=lr)
    plt.xlabel('Iterations')
    plt.ylabel('Mean Squared Error')
    plt.legend()


# In[24]:


updated_df


# ## Test Data

# In[25]:


learning_rate = [0.01]
for lr in learning_rate:
    MSE, count, beta, bias, array_MSE, array_iteration = gradient_descent(X_test,Y_test,beta,bias,lr,0.00000001)
    print(lr, MSE)
    plt.plot(array_iteration,array_MSE,label=lr)
    plt.xlabel('Iterations')
    plt.ylabel('Mean Squared Error')
    plt.legend()


# # Part 4 Handpicked 8 Features Selected

# In[26]:


round(project_df.corr(),5)


# In[27]:


selected_df = normalized_df.drop(columns=['STRN','KWI','MDIMA','KWG','SB','NDIMB'])


# In[28]:


training_set, test_set = train_test(selected_df)


# In[29]:


X_train = training_set.iloc[:,:-1]
X_test = test_set.iloc[:,:-1]
Y_train = training_set['Run_Avg']
Y_test = test_set['Run_Avg']


# ## Train Data

# In[30]:


for lr in learning_rate:
    beta = np.c_[np.zeros(X_train.shape[1])]
    beta.shape=np.squeeze(beta).shape

    bias = np.c_[np.zeros(1)]
    bias.shape=np.squeeze(bias).shape
    
    MSE, count, beta, bias, array_MSE, array_iteration = gradient_descent(X_train,Y_train,beta,bias,lr,0.0000001)
    print(lr, MSE)
    plt.plot(array_iteration,array_MSE,label=lr)
    plt.xlabel('Iterations')
    plt.ylabel('Mean Squared Error')
    plt.legend()


# In[31]:


selected_df


# ## Test Data

# In[32]:


for lr in learning_rate:
    MSE, count, beta, bias, array_MSE, array_iteration = gradient_descent(X_test,Y_test,beta,bias,lr,0.00000001)
    print(lr, MSE)
    plt.plot(array_iteration,array_MSE,label=lr)
    plt.xlabel('Iterations')
    plt.ylabel('Mean Squared Error')
    plt.legend()


# In[ ]:




