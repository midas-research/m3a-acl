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

def getSentences(text):
  ar = text.decode().split("\n\n")
  sentences = []
  boo = False
  for i in ar:
    if len(i)>=2 and len(i)//3>i.count("\n"):
      s = i.replace("\n"," ")
      if s == "Presentation" or s == "MANAGEMENT DISCUSSION SECTION":
        boo = True
        continue
      if boo and s[:4]!="Page" and "Company Name:" not in s[:14]:
        sentences.append(s)
  return sentences

YVals = pd.read_csv("Y_Volatility.csv")
files = YVals['File Name']

def cleanSent(sent):
  table = str.maketrans('', '', string.punctuation)
  sent = [s.translate(table) for s in sent]
  sent = [s.lower() for s in sent]
  return numpy.array(sent)

from sentence_transformers import SentenceTransformer
model = SentenceTransformer('bert-base-uncased')

index = 1
for i in files:
  print(index,i)
  df = pd.read_csv(i[:-4]+"_Annotations.csv")
  sent = df['Sentences']
  sent = cleanSent(sent)
  xEmb = model.encode(sent)
  sent = sent.reshape((-1,1))
  ar = numpy.concatenate((sent,xEmb),axis=1)
  df = pd.DataFrame(ar)
  df.to_csv(i[:-4]+"_BertBaseEmbeddings.csv")
  index += 1