
# coding: utf-8

# In[2]:

import sys						# System-specific parameters and functions
import csv						#Using the CSV module in Python
import tweepy					# enables Python to communicate with Twitter platform and use its API.
import pandas as pd				+#python library basically used for data analysis.
import numpy as np				#NumPy is the fundamental package for scientific computing with Python.
import os						#OS stands for operating system
from PIL import Image			#Python image library
import re						#regular expression
from ast import literal_eval	#evaluate an expression node or a Unicode or Latin-1 encoded string containing a Python literal or container display
import pyodbc					#Python ODBC - connecting to server


# In[3]:

from nltk.corpus import stopwords   						# stopwords : 'the', 'is', 'are'
from os import path											# importing path
from scipy.misc import imread								# imread uses the Python Imaging Library (PIL) to read an image
import collections											# high performance container data type
import operator												# standard operators as functions
import matplotlib.pyplot as plt								# collection of command style functions that make matplotlib work like MATLAB
#get_ipython().magic('matplotlib inline')					# To enable the inline backend for usage with the IPython Notebook.
from wordcloud import WordCloud, STOPWORDS					# 
from nltk.stem import PorterStemmer							# Stemming the words and removing the tenses.
from nltk.tokenize import sent_tokenize, word_tokenize		# divides a string into substrings by splitting on specified string


# In[4]:

def remove_stopwords(lis):                               # removing all the stop words from the list of tweets
    ii = 0
    while ii < len(lis):
        if lis[ii] in stopwords.words('english') or lis[ii] == 'http' or len(lis[ii])<4:   # from nltk.corpus and for word of lenth less than 4
            lis.remove(lis[ii])
        else:
            ii+=1
    return lis


# In[5]:
source ='/Users/jayasrisanthappan/Documents/Official/Step1_ Topic Modelling/Data/'

df = pd.read_excel(source+"Week_wise_Tweets_BankA.xlsx")                                                        # get the data set
df_c = df.set_index(pd.to_datetime(df[2])).drop(2, axis=1)                                                      # grouping the data weekly bases
df_c = df_c.to_period(freq='w')																					# setting the period to week.
df_c.index = list(map(str,df_c.index))


# In[6]:

s = 0
e = df.shape[0]
pro = [[]]
for row in range(len(df.iloc[s:e,3])):                                                                          # for all the tweets in the data set
    pro.append([])
    pro[row-s] = remove_stopwords(literal_eval(df.iloc[row,3]))                                                 # remove the stop words
    print("{0}%\r".format(int((row-s+1)/(e-s)*100)),end='')


# In[354]:

pro.pop()
df_c.iloc[s:e,2] = pro
df_c.to_excel(source+"Week_Filtered_Tweets_BankA.xlsx")                                                         # save the filtered file to excel file


# In[4]:

df_c = pd.read_excel(source+"Week_Filtered_Tweets_BankA.xlsx")
dff = df_c
dates_list = dff.index.duplicated(keep='first')                                                                 # generate the unique week lable from the data set
dates_list = np.logical_not(list(dates_list))												# save other than duplicated
dates_list = dff.index[dates_list]
dates_list = list(map(str,dates_list))																			# mapping according to week


#                                                                         ### Most popular Words

# In[5]:

popular_words = pd.DataFrame()
popular_words[0] = None
dff.index = list(map(str,dff.index))                                                                            # formating the index of the dataframe


# In[6]:

count = 0
for dat in dates_list:
    popular_words = popular_words.reindex(popular_words.index.union(str(dat).split(' ')))                       # splitting popular words 
    try:
        tweets = literal_eval(dff.loc[dat][3])                                                                  # generate the list of all the tweets for the week
    except ValueError as e:																						# created exception object 
        tweets = literal_eval(list(dff.loc[dat][3])[0])															# making list of row 3 and col 1
        for my_t in range(1, len(list(dff.loc[dat][3]))):														# from 1 to length of list
            tweets = tweets + literal_eval(list(dff.loc[dat][3])[my_t])											#    append
    count+=1
    top = collections.Counter(tweets)																			# forming a data structure and storing the number of tweets
    popular_words[0][dat] = top.most_common()[:10]                                                              # generating top words for each week
    print("{0}%\r".format(int(((count+1)/(len(dates_list)))*100)),end='')										# formating the output


# In[7]:

popular_words.to_excel(source+"BankA_Week_Wise_Top_Words.xlsx")                                                     # store the top tweets into a file
popular_words.index = list(map(str,popular_words.index))														# making list of indices


# In[14]:

hist_df = pd.DataFrame(index = dates_list[-5:])                                                                 # data-frame will have all the top tweets words for last month
hist_df[0] = hist_df[1] = hist_df[2] = hist_df[3] = hist_df[4] = hist_df[5] = None
hist_df[6] = hist_df[7] = hist_df[8] = hist_df[9] = None
hist_df.columns = list(pd.DataFrame(popular_words.loc[dates_list[-1]][0]).set_index(0).index)[:10]


# In[15]:

for date in dates_list[-5:]:                                                                                    # filling the hist_df values counting from right
    plt_df = pd.DataFrame(popular_words.loc[date][0]).set_index(0)												# set index to zero then store
    words = plt_df.to_dict()[1]																					# converting dataframe to directory
    plt_df.plot.bar()																							#   Plotting
    plt.title("Bar Plot [ Week : {0} ]".format(date))                                                           # plotting the final dataframe
    plt.show()																									# showing the plot
    dat = date
    for x in list(plt_df.index):
        if x in hist_df.columns:
            hist_df[x][date] = plt_df[1][x]
    wordcloud = WordCloud(background_color='white',                                                             # generate the word cloud for the top words
                      relative_scaling = 1.0,
                      width=1000, height=500,
                      stopwords = stopwords.words('english')                                                    # set or space-separated string
                      ).generate_from_frequencies(words)
    fig = plt.figure(figsize=(8,5))
    ax = fig.add_subplot(2,1,1)
    ax.imshow(wordcloud)
    ax.axis("off")
    fig.show()


# In[22]:

hist_df.plot.bar(figsize=(18,6))
plt.xticks(rotation=0)
plt.xlabel('Dates')
plt.ylabel('Market Cap')
plt.title('Top Topics in last 5 Weeks')
plt.show()

