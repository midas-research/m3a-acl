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

smile = opensmile.Smile(
    feature_set=opensmile.FeatureSet.GeMAPSv01b,
    feature_level=opensmile.FeatureLevel.Functionals,
)

index = 1
for f in files:
  text = textract.process(f)
  sentences = getSentences(text)[:-1]
  sent = []
  for i in range(len(sentences)):
    if sentences[i].count(" ")>=9:
      sent.append(sentences[i])

  song = AudioSegment.from_mp3(f[:-4] + ".mp3")
  song.export("copy.mp3", format="mp3")
  
  fi = open("text.txt",'w')
  for i in sent:
    fi.write(i)
    fi.write("\n")
  fi.close()

  config_string = u"task_language=eng|is_text_type=plain|os_task_file_format=json"
  task = Task(config_string=config_string)
  task.audio_file_path_absolute = "copy.mp3"
  task.text_file_path_absolute = u"/content/text.txt"
  task.sync_map_file_path_absolute = u"/content/mapping.json"
  ExecuteTask(task).execute()
  task.output_sync_map_file()
  
  fi = open("/content/mapping.json")
  data = json.load(fi)

  audioEmb = []
  for i in range(len(sent)):
    begin = int(float(data['fragments'][i]['begin']))*1000
    end = int(float(data['fragments'][i]['end']))*1000
    s = song[begin:end+1]
    s.export("sentAudio.mp3",format="mp3")
    x = smile.process_file("sentAudio.mp3")
    audioEmb.append(x.to_numpy().reshape((62,)))
  
  audioEmb = numpy.array(audioEmb)
  sent = numpy.array(sent)
  sent = sent.reshape((-1,1))
  ar = numpy.concatenate((sent,audioEmb),axis=1)
  df = pd.DataFrame(ar)
  df.to_csv(f[:-4]+"_AudioEmbeddings.csv")
  
  print(index,f)
  index += 1
