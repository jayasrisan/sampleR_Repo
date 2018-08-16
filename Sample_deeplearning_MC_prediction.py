
# coding: utf-8

# In[1]:

import numpy as np                                                           # linear algebra
import pandas as pd                                                  # data processing, CSV file I/O (e.g. pd.read_csv)
from subprocess import check_output
from keras.layers.core import Dense, Activation, Dropout
from keras.layers.recurrent import LSTM
from keras.models import Sequential
from sklearn.cross_validation import  train_test_split
import time                                                            #helper libraries
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
get_ipython().magic('matplotlib inline')
from numpy import newaxis
from sklearn.metrics import mean_squared_error


# In[2]:

                                                                                  # convert an array of values into a dataset matrix
def create_dataset(dataset, look_back=1):
	dataX, dataY = [], []
	for i in range(len(dataset)-look_back-1):                             # For all the value in the dataset that need to be processed 
		a = dataset[i:(i+look_back), 0]                     #       create a common 1-D array of values with 'lookup'
		dataX.append(a)                                       #       to find the similarity between the word and the previous word
		dataY.append(dataset[i + look_back, 0])
	return np.array(dataX), np.array(dataY)


# In[3]:

def plot_results_multiple(predicted_data, true_data,length):                  # Create plot of the test dataset 
    plt.plot(scaler.inverse_transform(true_data.reshape(-1, 1))[length:])          # to check the trend in the dataset
    plt.plot(scaler.inverse_transform(np.array(predicted_data).reshape(-1, 1))[length:])
    plt.show()
    
                                                                          # predict lenght consecutive values from a real one
def predict_sequences_multiple(model, firstValue,length):
    prediction_seqs = []
    curr_frame = firstValue
    
    for i in range(length):                                        # for 'length' (i.e. no of predictions that need to be done on dataset)
        predicted = []                                                                                                                          
        
        predicted.append(model.predict(curr_frame[newaxis,:,:])[0,0])         # using predict to find the best axis for the data set
        
        curr_frame = curr_frame[0:]
        curr_frame = np.insert(curr_frame[0:], i+1, predicted[-1], axis=0)  # inserting the non-scalar predicted value for current dataframe
                                                                    #       and then using the new dataframe for next prediction
        prediction_seqs.append(predicted[-1])
        
    return prediction_seqs                                   # Return the predicted sequence


# In[56]:

def predict(prices_dataset, p_len):                  # p_len is the Number of prediction that need to be made
    jpm_stock_prices = prices_dataset['StockPrice'].astype('float32')   # Read the dataframe and the convert all the values to 'float32'
    jpm_stock_prices = jpm_stock_prices.reshape(prices_dataset.shape[0], 1)  # reshape the dataframe to 1-D array
    plt.plot(jpm_stock_prices)
    plt.show()
    scaler = MinMaxScaler(feature_range=(0, 1))  # Creating the scalar transformation of the values so that the 
    jpm_stock_prices = scaler.fit_transform(jpm_stock_prices)                                                           #       predicted value can again be scaled back to know the exact value
    train_size = int(len(jpm_stock_prices) * 0.90)                                                                      # 90% of the data from the dataset will be taken as a training Data
    test_size = len(jpm_stock_prices) - train_size                                                                      # rest wil be test data
    train, test = jpm_stock_prices[0:train_size,:], jpm_stock_prices[train_size:len(jpm_stock_prices),:]                # split the dataset into train and test data
    look_back = 1
    trainX, trainY = create_dataset(train, look_back)                                                                   # create the train X and Y dataset
    testX, testY = create_dataset(test, look_back)
    trainX = np.reshape(trainX, (trainX.shape[0], 1, trainX.shape[1]))
    testX = np.reshape(testX, (testX.shape[0], 1, testX.shape[1]))
    
    model = Sequential()                                                                                                # create a new instance of the sequential data to predict

    model.add(LSTM(                                                                                                     # adding the LSTM model to the sequence
        input_dim=1,                                                                                                    #       since we have 1-D array as input
        output_dim=50,
        return_sequences=True))
    model.add(Dropout(0.2))     # The maximum loss that can be ignored
    model.add(LSTM(          # 2nd instance of the LSTM Model 
        100,                     # output of 1st model will be input for the other
        return_sequences=False))
    model.add(Dropout(0.2))
    model.add(Dense(
        output_dim=1))
    model.add(Activation('linear'))              # adding 'linear' ACTIVATION
    start = time.time()                         # to calculate the total time taken
    model.compile(loss='mse', optimizer='rmsprop')      # setting loss variable as 'mse' and using 'rmsprop' optimizer
    
    model.fit(                                           # fitting the training dataset to the model
        trainX,                                      #       with a batch of 128 and running it for 10 'epochs' 
        trainY,
        batch_size=128,
        epochs=10,
        validation_split=0.05)
    
    predict_length=p_len                           # return the predicted value along with the test values
        
    return scaler.inverse_transform(np.array(predict_sequences_multiple(model, testX[0], predict_length)).reshape(-1, 1)), scaler.inverse_transform(np.array(testY).reshape(-1, 1))


#                                                                                                                       ## Subject 1

# In[63]:
source ="/Users/jayasrisanthappan/Documents/Official/Step2_ Prediction using Deep learning/Step2a_input data/"
prices_dataset =  pd.read_csv(source+'BankB_6years.csv')                                                                         # read the dataset 
monthly_df = prices_dataset.set_index(pd.to_datetime(prices_dataset['Date'])).drop('Date', axis=1)                      # chance the frequecy of the data to month wise and assign it to new
monthly_df = monthly_df.to_period(freq='m')                                                                             #       data frame
monthly_df.index = list(map(str,monthly_df.index))
monthly_df = monthly_df.reset_index().drop_duplicates(subset='index', keep='first').set_index('index')                  # get all the diffrent months tha tare available


# In[65]:

prices_dataset = prices_dataset.set_index('Date')                                                                       # set date column as the new indes


#                                                                                                                       ## Subject 2

# In[64]:

prices_dataset_j =  pd.read_csv(source+'BankA_6years.csv')                                                                       # same as 'Subject 1'
monthly_df_j = prices_dataset_j.set_index(pd.to_datetime(prices_dataset_j['Date'])).drop('Date', axis=1)
monthly_df_j = monthly_df_j.to_period(freq='m')
monthly_df_j.index = list(map(str,monthly_df_j.index))
monthly_df_j = monthly_df_j.reset_index().drop_duplicates(subset='index', keep='first').set_index('index')


# In[66]:

prices_dataset_j = prices_dataset_j.set_index('Date')


#                                                                                                                       ## Prediction

#                                                                                                                               ###### Monthly

# In[77]:

predict_length = 3
plt_df = pd.DataFrame(index=list(map(str,list(monthly_df.index[-predict_length:]))))                                    # create a blank Dataframe which will have predicted and actual value 
plt_df[0] = plt_df[1] = plt_df[2] = plt_df[3] = None                                                                    #       for both 'bank 1' and 'bank 2'
header = [ np.array(['BankA','BankA','BankB','BankB']),
         np.array(['Prediction','Actual','Prediction','Actual'])]
pred , testY = predict(monthly_df,3)                                                                                    # predicting the value for bank 1
plt_df[0] = pred
plt_df[1] = testY[0:predict_length]
pred , testY = predict(monthly_df_j,3)                                                                                  # Predicting the value for bank 2
plt_df[2] = pred
plt_df[3] = testY[0:predict_length]
plt_df.columns = header


#                                                                                                                       ### Monthly Prediction

# In[79]:

print(plt_df)                                                                                                           # FINAL DATAFRAME
plt_df.plot(figsize=(15,5),title="Monthly Prediction")                                                                  # plotting final dataframe
ind = list(plt_df.index)
plt.xlabel("{0}\t{1}\t{2}".format(ind[0],ind[1],ind[2]).replace('\t',' '*80))
plt.ylabel("Market Cap")
plt.show()


#                                                                                                                       ### Weekly Prediction

# In[80]:

predict_length=4
plt_df = pd.DataFrame(index=list(map(str,list(prices_dataset.index[-predict_length:]))))                                # final dataframe of prediction and actual for weekly data
plt_df[0] = plt_df[1] = plt_df[2] = plt_df[3] = None
header = [ np.array(['BankA','BankA','BankB','BankB']),
         np.array(['Prediction','Actual','Prediction','Actual'])]
pred , testY = predict(prices_dataset,4)                                                                                # Predict the value for all the 4 weeks of the latest month 'bank 1'
plt_df[0] = pred
plt_df[1] = testY[0:predict_length]
pred , testY = predict(prices_dataset_j,4)                                                                              # Predict the value for all the 4 weeks of the latest month 'bank 2'
plt_df[2] = pred
plt_df[3] = testY[0:predict_length]
plt_df.columns = header


# In[83]:

print(plt_df)
plt_df.plot(figsize=(15,5),title="Weekly Prediction",x = plt_df.index)                                                  # plotting the final dataframe of weekly data prediction
ind = list(plt_df.index)
plt.xlabel("{0}\t{1}\t{2}\t{3}".format(ind[0],ind[1],ind[2],ind[3]).replace('\t',' '*65))
plt.ylabel("Market Cap")
plt.show()


# In[84]:

predi = pred[2]
testy = testY[0:predict_length][2]
print("RMSE : ",np.sqrt(mean_squared_error(predi,testy)))                                                               # to get the RMSE value of the prediction

