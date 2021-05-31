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

def getSpeakers(text):
  ar = text.decode().split("\n")
  sentences = []
  for i in ar:
    if len(i)>=2:
      sentences.append(i)

  ind = -1
  for i in range(len(sentences)):
    s = sentences[i].strip()
    if s[-12:] == "Participants" and ind == -1:
      ind = i
    elif s == "Presentation" or s == "MANAGEMENT DISCUSSION SECTION":
      ind1 = i
      break

  speakerList = []
  for i in sentences[ind+1:ind1]:
    if i[-12:] != "Participants":
      speakerList.append(i)
  for i in range(len(speakerList)):
    if "," in speakerList[i]:
      ar = speakerList[i].split(",")
      speakerList[i] = ar[0].strip()
  return speakerList

def annotateSentences(sentences, speakers):
  annotations = [-1]*len(sentences)
  for i in range(len(speakers)):
    for j in range(len(sentences)):
      if speakers[i] in sentences[j][:len(speakers[i])+10]:
        annotations[j] = i
  prev = 0
  for i in range(len(annotations)):
    if annotations[i] == -1:
      annotations[i] = prev
    else:
      prev = annotations[i]
  return annotations

for index in range(len(files)):
  f = files[index]
  print(index,f)
  text = textract.process(f)
  speakerList = ["Operator"] + getSpeakers(text)
  speakerInd = [i for i in range(len(speakerList))]
  sentences = getSentences(text)[:-1]
  annotations = annotateSentences(sentences, speakerList)
  sent = []
  ann = []
  for i in range(len(sentences)):
    if sentences[i].count(" ")>=9:
      sent.append(sentences[i])
      ann.append(annotations[i])

  dic = {"Speaker Name":speakerList}
  speakerDF = pd.DataFrame(dic)
  speakerDF.to_csv(f[:-4]+"_Speakers.csv")
  
  dic = {"Sentences":sent,"Annotation":ann}
  annotDF = pd.DataFrame(dic)
  annotDF.to_csv(f[:-4]+"_Annotations.csv")