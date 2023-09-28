#!/usr/bin/env python
# coding: utf-8

# # Stock Prediction

# In[107]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import seaborn as sns
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
import math
from sklearn.metrics import mean_squared_error


# In[108]:


#Get the Dataset 
a =pd.read_csv("T:\\Internship\\Bharat intern\\Stock prediction\\all_stocks_5yr.csv")


# In[109]:


df = pd.DataFrame(a)


# In[110]:


df=df.tail(50000)


# In[111]:


df.head()


# In[112]:


df.shape


# In[113]:


df.dtypes


# In[114]:


df.describe()


# In[115]:


df.info()


# In[116]:


df.isnull().sum()


# In[117]:


df.duplicated().any(axis=0)


# In[118]:


df1= df.reset_index()['close']


# In[119]:


df1.head()


# In[120]:


df1.shape


# In[121]:


df1.head()


# In[122]:


plt.plot(df1)
plt.show()


# In[123]:


scaler = MinMaxScaler(feature_range =(0,1))
df1 = scaler.fit_transform(np.array(df1).reshape(-1,1))


# In[124]:


df1


# In[125]:


len(df1)


# In[129]:


#splitting dataset into train and test split
training_size = int(len(df1)*0.65)
test_size = len(df1)-training_size
train_data, test_data= df1[0:training_size,:],df1[training_size:len(df1),:1]


# In[130]:


training_size,test_size


# In[131]:


train_data


# In[132]:


plt.plot(train_data)
plt.show()


# In[ ]:


test_data


# In[133]:


plt.plot(test_data)
plt.show


# In[135]:


#convert an array of values into a dataset matrix
def create_dataset(dataset,time_step=1):
    dataX, dataY=[],[]
    for i in range(len(dataset)-time_step-1):
        a = dataset[i:(i+time_step),0]   #i=0, X=0
        dataX.append(a)
        dataY.append(dataset[i +time_step,0])
    return np.array(dataX), np.array(dataY)


# In[140]:


#reshape into X=t, t+1, t+2..t+99 and Y= t+100
time_step = 100
X_train, y_train = create_dataset(train_data, time_step)

X_test,ytest = create_dataset(test_data, time_step)


# In[141]:


print(X_train.shape),print(y_train.shape)


# In[142]:


print(X_test.shape),print(ytest.shape)


# In[143]:


#reshape input to be [samples, time steps, feature] which is required for LSTM
X_train = X_train.reshape(X_train.shape[0],X_train.shape[1],1)
X_test = X_test.reshape(X_test.shape[0],X_test.shape[1],1)


# In[152]:


from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import LSTM


# In[153]:


model=Sequential()
model.add(LSTM(50,return_sequences=True,input_shape=(100,1)))
model.add(LSTM(50,return_sequences=True))
model.add(LSTM(50))
model.add(Dense(1))
model.compile(loss='mean_squared_error',optimizer='adam')


# In[154]:


model.summary()


# In[197]:


model.fit(X_train,y_train,validation_data=(X_test,ytest),epochs=100,batch_size= 64,verbose=1)


# In[159]:


#Lets do the prediction and check performance metrics
train_predict= model.predict(X_train)
test_predict=model.predict(X_test)


# In[160]:


#transformaback to original form
train_predict= scaler.inverse_transform(train_predict)
test_predict=scaler.inverse_transform(test_predict)


# In[161]:


#calculate RMSE peformance metrics
math.sqrt(mean_squared_error(y_train,train_predict))


# In[ ]:


#Test DATA RMSE


# In[162]:


math.sqrt(mean_squared_error(ytest,test_predict))


# In[164]:


#plotting 
#shift train prediction for plotting
look_back= 100
trainPredictPlot = np.empty_like(df1)
trainPredictPlot[:, :]=  np.nan
trainPredictPlot[look_back:len(train_predict)+look_back,:]= train_predict
#shift test prediction for plotting
testPredictPlot= np.empty_like(df1)
testPredictPlot[:, :]= np.nan
testPredictPlot[len(train_predict)+(look_back*2)+1:len(df1)-1,:]= test_predict
#plot baseline and prediction
plt.plot(scaler.inverse_transform(df1))
plt.plot(trainPredictPlot)
plt.plot(testPredictPlot)
plt.show()


# In[165]:


len(test_data)


# In[168]:


x_input = test_data[340:].reshape(1,-1)
x_input.shape


# In[170]:


temp_input = list(x_input)
temp_input= temp_input[0].tolist()


# In[171]:


temp_input


# In[177]:


#demonstrate prediction for next 30 day
from numpy import array

lst_output=[]
n_steps=100
i=0
while(i<30):
    if(len(temp_input)>100):
        x_input=np.array(temp_input[1:])
        print('{} day input {}'.format(i,x_input))
        x_input = x_input.reshape(1,-1)
        x_input = x_input[:100].reshape((1, 17159,1))
        yhat =  model.predict(x_input,verbose=0)
        print('{} day output {}'.format(i,yhat))
        temp_input.extend(yhat[0].tolist())
        temp_input = temp_input[1:]
        lst_output.extend(yhat.tolist())
        i=i+1
    else:
        x_input = x_input.reshape((1, n_steps,1))
        yhat =  model.predict(x_input,verbose=0)
        print(yhat[0])
        temp_input.extend(yhat[0].tolist())
        print(len(temp_input))
        lst_output.extend(yhat.tolist())
        i=i+1
        
print(lst_output)


# In[178]:


day_new=np.arange(1,101)
day_pred=np.arange(101,131)


# In[179]:


len(df1)


# In[180]:


scaler.inverse_transform(lst_output)


# In[187]:


plt.plot(day_new,scaler.inverse_transform(df1[2250:2350]))
plt.plot(day_pred[:19],scaler.inverse_transform(lst_output))
plt.savefig('15month.png')


# In[194]:


df3 = df1.tolist()
df3.extend(lst_output)
plt.plot(df3[1000:1500])
plt.show()


# In[195]:


df3= scaler.inverse_transform(df3).tolist()


# In[196]:


plt.plot(df3)
plt.show()


# In[ ]:



