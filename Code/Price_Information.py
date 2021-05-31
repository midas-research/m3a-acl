from google.colab import drive
drive.mount('/content/drive')

!pip install textract
!pip install yahoofinancials
!apt-get install -qq libespeak-dev > /dev/null
!pip install -q https://codeload.github.com/readbeyond/aeneas/zip/devel
!pip install pydub
!pip install opensmile
!git clone git://git.code.sf.net/p/sox/code sox
!apt -qq install -y sox
!pip install -U sentence-transformers

from aeneas.executetask import ExecuteTask
from aeneas.task import Task
import json
import textract
import re
import statistics
import os
from tqdm.auto import tqdm
import pandas as pd
from yahoofinancials import YahooFinancials as yf
import datetime
import numpy
from pydub import AudioSegment
import os
import opensmile
import string

YVals = pd.read_csv("Y_Volatility.csv")
files = YVals['File Name']

lenCN = len("Company Name: ")
lenCT = len("Company Ticker: ")
lenD = len("Date: ")

companyName = []
companyTicker = []
date = []
i = 1
for f in files:
  text = textract.process(f)
  ar = text.decode().split("\n\n")
  header = ar[0].split("\n")
  companyName.append(header[0][lenCN:])
  companyTicker.append(header[1][lenCT:])
  date.append(header[2][lenD:])  
  print(i,f,[companyName[-1],companyTicker[-1],date[-1]])
  i += 1

dic = {"File Name":files,"Company Name":companyName,"Company Ticker":companyTicker,"Date":date}
priceDF = pd.DataFrame(dic)
priceDF.to_csv("PriceData.csv")

"""# Get Price Information

"""

data = []
priceDF = pd.read_csv("PriceData.csv")
for i in range(len(priceDF)):
  stock = priceDF.loc[i]
  ticker = stock["Company Ticker"]
  if type(ticker)==str and " " in ticker:
    ticker = ticker.split(" ")[0]
    if ticker.isalpha():
      data.append([stock["File Name"],ticker,stock["Date"]])
    else:
      print(stock["File Name"],stock["Company Ticker"],ticker,ticker.isalpha(),len(ticker))

column_ind = []
for i in range(61):
  column_ind.append("Day"+str((i-30))+"_Date")
  column_ind.append("Day"+str((i-30))+"_adjclose")

row_ind = []
stocks = []

for i in tqdm(range(len(data))):
  stock = data[i]
  print(stock)
  name = stock[0]
  year, month, day = map(int,stock[2].split("-"))
  date = datetime.datetime(year,month,day)

  if((datetime.datetime.today()-date).days>56):

    dateBeg = date + datetime.timedelta(days=-56)
    yearB = dateBeg.strftime('%Y')
    monthB = dateBeg.strftime('%m')
    dayB = dateBeg.strftime('%d')
    dateBeg = yearB + "-" + monthB + "-" + dayB

    dateEnd = date + datetime.timedelta(days=56)
    yearE = dateEnd.strftime('%Y')
    monthE = dateEnd.strftime('%m')
    dayE = dateEnd.strftime('%d')
    dateEnd = yearE + "-" + monthE + "-" + dayE

    ticker = stock[1]
    ytick = yf(ticker)
    prices = ytick.get_historical_price_data(dateBeg,dateEnd,time_interval='daily')

    if(prices[ticker] != None):
      if('prices' in prices[ticker].keys() and prices[ticker]['prices']!=None):
        df = pd.DataFrame(prices[ticker]['prices'])

        mid = 0;

        while(mid<df.shape[0] and str(date.date()) != df.iloc[mid,7]):
          mid+=1;

        if(mid<df.shape[0]):

          df = df.iloc[mid-30:mid+31,6:]
          stock = []
          for j in range(df.shape[0]):
            stock.append(df.iloc[j,1])
            stock.append(df.iloc[j,0])

          row_ind.append(name)
          stocks.append(stock)

Stock_Info = pd.DataFrame(data=stocks,index = row_ind, columns=column_ind)
Stock_Info.to_csv('Stock_Info.csv')

"""# Stock Volatility

"""

def getVolatility(df,index,tau,priceColumn):
  mid = len(priceColumn)//2 # Day 0
  
  if tau<0:
    start = mid+tau-1
    end = mid
    
  else:
    start = mid-1
    end = mid+tau
  
  ar = []
  for i in range(start,end+1):
    ar.append(df.loc[index][priceColumn[i]])
  
  arr = []
  for i in range(1,len(ar)):
    arr.append((ar[i]-ar[i-1])/float(ar[i-1]))
  
  vol = numpy.std(arr)
  vol = numpy.log(vol)
  return -vol

stockInfo = pd.read_csv("Stock_Info.csv")
priceCol = stockInfo.columns[3::2]

files = []
vFuture3 = []
vFuture7 = []
vFuture15 = []
dates = []
for i in range(len(stockInfo)):
  files.append(stockInfo.loc[i]['Unnamed: 0.1'])
  vFuture3.append(getVolatility(stockInfo,i,3,priceCol))
  vFuture7.append(getVolatility(stockInfo,i,7,priceCol))
  vFuture15.append(getVolatility(stockInfo,i,15,priceCol))
  dates.append(stockInfo.loc[i]['Day0_Date'])
  if numpy.isnan(vPast15[-1]) or numpy.isinf(vPast15[-1]):
    print(i,stockInfo.loc[i])
    files = files[:-1]
    vFuture3 = vFuture3[:-1]
    vFuture7 = vFuture7[:-1]
    vFuture15 = vFuture15[:-1]
    dates = dates[:-1]

dic = {"File Name":files,"Date":dates,"vFuture3":vFuture3,"vFuture7":vFuture7,"vFuture15":vFuture15}
df = pd.DataFrame(dic)
df.to_csv("Y_Volatility.csv")

"""# Price Movement Direction

"""

df = pd.read_csv("Stock_Info.csv")
files = []
YT3 = []
YT7 = []
YT15 = []

for i in range(len(df)):
  true0 = df.loc[i]['Day0_adjclose']
  true3 = df.loc[i]['Day3_adjclose']
  true7 = df.loc[i]['Day7_adjclose']
  true15 = df.loc[i]['Day15_adjclose']
  
  files.append(df.loc[i]['Unnamed: 0.1'])
  
  if true3>true0:
    YT3.append(1)
  else:
    YT3.append(0)
  
  if true7>true0:
    YT7.append(1)
  else:
    YT7.append(0)
  
  if true15>true0:
    YT15.append(1)
  else:
    YT15.append(0)

dic = {"File Name":files,"YT3":YT3,"YT7":YT7,"YT15":YT15}
priceDF = pd.DataFrame(dic)
priceDF.to_csv("Y_UD_Cleaned.csv")
