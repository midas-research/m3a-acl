import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import load_model
from tensorflow.keras.layers import *

from tqdm.auto import tqdm
from sklearn.metrics import *
import pandas as pd
import numpy as np
import math
# %matplotlib inline
import matplotlib.pyplot as plt
import gc

#Hyper Parameters
batch_size = 64
learning_rate = 0.001

volatility_feedforward_size = 16
volatility_hidden_dim = 16
volatility_dropout = 0.1

movement_feedforward_size = 64
movement_hidden_dim = 32
movement_dropout = 0.0

"""# M3A Function Declaration"""

class MultiHeadSelfAttention(layers.Layer):
    def __init__(self, embed_dim, num_heads=8,**kwargs):
        super(MultiHeadSelfAttention, self).__init__(**kwargs)
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        if embed_dim % num_heads != 0:
            raise ValueError(
                f"embedding dimension = {embed_dim} should be divisible by number of heads = {num_heads}"
            )
        self.projection_dim = embed_dim // num_heads
        self.query_dense = Dense(embed_dim)
        self.key_dense = Dense(embed_dim)
        self.value_dense = Dense(embed_dim)
        self.combine_heads = Dense(embed_dim)

    def get_config(self):
        config = super().get_config().copy()
        config.update({
            'embed_dim': self.embed_dim,
            'num_heads': self.num_heads,
        })
        return config

    def attention(self, query, key, value):
        score = tf.matmul(query, key, transpose_b=True)
        dim_key = tf.cast(tf.shape(key)[-1], tf.float32)
        scaled_score = score / tf.math.sqrt(dim_key)
        weights = tf.nn.softmax(scaled_score, axis=-1)
        output = tf.matmul(weights, value)
        return output, weights

    def separate_heads(self, x, batch_size):
        x = tf.reshape(x, (batch_size, -1, self.num_heads, self.projection_dim))
        return tf.transpose(x, perm=[0, 2, 1, 3])

    def call(self, inputs):
        batch_size = tf.shape(inputs)[0]
        query = self.query_dense(inputs)  # (batch_size, seq_len, embed_dim)
        key = self.key_dense(inputs)  # (batch_size, seq_len, embed_dim)
        value = self.value_dense(inputs)  # (batch_size, seq_len, embed_dim)
        query = self.separate_heads(
            query, batch_size
        )  # (batch_size, num_heads, seq_len, projection_dim)
        key = self.separate_heads(
            key, batch_size
        )  # (batch_size, num_heads, seq_len, projection_dim)
        value = self.separate_heads(
            value, batch_size
        )  # (batch_size, num_heads, seq_len, projection_dim)
        attention, weights = self.attention(query, key, value)
        attention = tf.transpose(
            attention, perm=[0, 2, 1, 3]
        )  # (batch_size, seq_len, num_heads, projection_dim)
        concat_attention = tf.reshape(
            attention, (batch_size, -1, self.embed_dim)
        )  # (batch_size, seq_len, embed_dim)
        output = self.combine_heads(
            concat_attention
        )  # (batch_size, seq_len, embed_dim)
        return output

class TransformerBlock(layers.Layer):
    def __init__(self, embed_dim, num_heads, ff_dim, rate=0.1,**kwargs):
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.ff_dim = ff_dim
        super(TransformerBlock, self).__init__(**kwargs)
        self.att = MultiHeadSelfAttention(self.embed_dim, self.num_heads)
        self.ffn = keras.Sequential(
            [Dense(self.ff_dim, activation="relu"), Dense(self.embed_dim),]
        )
        self.layernorm1 = LayerNormalization(epsilon=1e-6)
        self.layernorm2 = LayerNormalization(epsilon=1e-6)
        self.dropout1 = Dropout(rate)
        self.dropout2 = Dropout(rate)

    def get_config(self):
        config = super().get_config().copy()
        config.update({
            'embed_dim': self.embed_dim,
            'num_heads': self.num_heads,
            'ff_dim': self.ff_dim
        })
        return config

    def call(self, inputs, training):
        attn_output = self.att(inputs)
        attn_output = self.dropout1(attn_output, training=training)
        out1 = self.layernorm1(inputs + attn_output)
        ffn_output = self.ffn(out1)
        ffn_output = self.dropout2(ffn_output, training=training)
        return self.layernorm2(out1 + ffn_output)

def createModelV(emd1, emd2, heads, dimFF, dimH, drop, maxlen, maxSpeaker):
  embed_dim1 = emd1   # Embedding size for Text 
  embed_dim2 = emd2   # Embedding size for Audio
  num_heads = heads   # Number of attention heads
  ff_dim = dimFF      # Hidden layer size in feed forward network inside transformer
  hidden_dim = dimH   # Hidden layer Dimension
  dropout = drop      # Dropout

  text = Input(shape=(maxlen,embed_dim1))
  audio = Input(shape=(maxlen,embed_dim2))
  pos = Input(shape=(maxlen,embed_dim2))
  speak = Input(shape=(maxLen,maxSpeaker+1))

  newtext = TimeDistributed(Dense(62))(text)

  attentionText2 = TimeDistributed(Dense(62, activation='softmax'))(newtext)
  attentionAudio2 = TimeDistributed(Dense(62, activation='softmax'))(audio)
  attentionSum = attentionText2+attentionAudio2
  attentionText2 = attentionText2/attentionSum
  attentionAudio2 = attentionAudio2/attentionSum

  attendedText = newtext*attentionText2 
  attendedAudio = audio*attentionAudio2 

  fused = attendedText*attentionText2 + attendedAudio*attentionAudio2 + pos
  fusedSpeaker = Concatenate(axis=2)([fused,speak])

  transformer_block = TransformerBlock(embed_dim2+maxSpeaker+1, num_heads, ff_dim, 0)
  x = transformer_block(fusedSpeaker)
  x = layers.GlobalAveragePooling1D()(x)
  x = layers.Dense(hidden_dim, activation="relu")(x)
  x = layers.Dropout(dropout)(x)
  outputs = layers.Dense(1, activation="relu")(x)

  model = keras.Model(inputs=[text,audio,pos,speak], outputs=outputs)
  return model

def createModelC(emd1, emd2, heads, dimFF, dimH, drop, maxlen, maxSpeaker):
  embed_dim1 = emd1   # Embedding size for Text 
  embed_dim2 = emd2   # Embedding size for Audio
  num_heads = heads   # Number of attention heads
  ff_dim = dimFF      # Hidden layer size in feed forward network inside transformer
  hidden_dim = dimH   # Hidden layer Dimension
  dropout = drop      # Dropout
  
  text = Input(shape=(maxlen,embed_dim1))
  audio = Input(shape=(maxlen,embed_dim2))
  pos = Input(shape=(maxlen,embed_dim2))
  speak = Input(shape=(maxLen,maxSpeaker+1))

  newtext = TimeDistributed(Dense(62))(text)

  attentionText1 = TimeDistributed(Dense(62, activation='softmax'))(newtext)
  attentionAudio1 = TimeDistributed(Dense(62, activation='softmax'))(audio)
  attentionText2 = TimeDistributed(Dense(62, activation='softmax'))(newtext)
  attentionAudio2 = TimeDistributed(Dense(62, activation='softmax'))(audio)
  attentionSum = attentionText2+attentionAudio2
  attentionText2 = attentionText2/attentionSum
  attentionAudio2 = attentionAudio2/attentionSum

  attendedText = newtext*attentionText2 
  attendedAudio = audio*attentionAudio2 

  fused = attendedText*attentionText2 + attendedAudio*attentionAudio2 + pos
  fusedSpeaker = Concatenate(axis=2)([fused,speak])

  transformer_block = TransformerBlock(embed_dim2+maxSpeaker+1, num_heads, ff_dim, 0)
  x = transformer_block(fusedSpeaker)
  x = layers.GlobalAveragePooling1D()(x)
  x = layers.Dense(hidden_dim, activation="relu")(x)
  x = layers.Dropout(dropout)(x)
  outputs = layers.Dense(1, activation="sigmoid")(x)

  model = keras.Model(inputs=[text,audio,pos,speak], outputs=outputs)
  return model

"""# M3A Data Loading and Processing"""

YVals = pd.read_csv("Y_Volatility.csv")
files = YVals['File Name']
dates = YVals['Date']

path_to_files = 'Dataset/' # insert path with last /

X = []
maxLen = 0
index = 1
for i in tqdm(range(len(files))):
  f = files[i][:-4]
  d = str(dates[i]).split('/')
  date = d[-1] + '-'
  if(len(d[0]) == 2):
    date += d[0] + '-'
  else:
    date += '0'+d[0]+'-'
  if(len(d[1]) == 2):
    date += d[1] 
  else:
    date += '0'+d[1]
  folder = path_to_files + f + '_' + date+'/'
  df = pd.read_csv(folder+"Text.csv")
  df = df.drop([df.columns[0], df.columns[1]],axis=1)
  xEmb = df.to_numpy()
  X.append(xEmb)
  maxLen = max(maxLen, xEmb.shape[0])
  # print(index,f)
  index += 1

for i in tqdm(range(len(X))):
  xEmb = X[i]
  pad = maxLen-xEmb.shape[0]
  if pad != 0:
    padding = np.zeros((pad,768))
    xEmb = np.concatenate((padding,xEmb),axis=0)
  X[i] = xEmb
X_text = np.array(X)

Xspeak = []
maxSpeaker = 0
index = 1
for i in tqdm(range(len(files))):
  f = files[i][:-4]
  d = str(dates[i]).split('/')
  date = d[-1] + '-'
  if(len(d[0]) == 2):
    date += d[0] + '-'
  else:
    date += '0'+d[0]+'-'
  if(len(d[1]) == 2):
    date += d[1] 
  else:
    date += '0'+d[1]
  folder = path_to_files + f + '_' + date+'/'
  df = pd.read_csv(folder+"Text.csv")
  speaker = df['Speaker']
  Xspeak.append(speaker)
  maxSpeaker = max(maxSpeaker,max(speaker))

for i in tqdm(range(len(Xspeak))):
  speaker = Xspeak[i]
  s = []
  for j in range(len(speaker)):
    temp = np.zeros((maxSpeaker+1,))
    temp[speaker[j]] = 1
    s.append(temp)
  s = np.array(s)
  pad = maxLen-speaker.shape[0]
  if pad != 0:
    padding = np.zeros((pad,maxSpeaker+1))
    s = np.concatenate((padding,s),axis=0)
  Xspeak[i] = s
Xspeak = np.array(Xspeak)

X = []
maxLen = 0
index = 1
for i in tqdm(range(len(files))):
  f = files[i][:-4]
  d = str(dates[i]).split('/')
  date = d[-1] + '-'
  if(len(d[0]) == 2):
    date += d[0] + '-'
  else:
    date += '0'+d[0]+'-'
  if(len(d[1]) == 2):
    date += d[1] 
  else:
    date += '0'+d[1]
  folder = path_to_files + f + '_' + date+'/'
  df = pd.read_csv(folder+"Audio.csv")
  df = df.drop([df.columns[0]],axis=1)
  xEmb = df.to_numpy()
  X.append(xEmb)
  maxLen = max(maxLen, xEmb.shape[0])
  index += 1

for i in tqdm(range(len(X))):
  xEmb = X[i]
  pad = maxLen-xEmb.shape[0]
  if pad != 0:
    padding = np.zeros((pad,62))
    xEmb = np.concatenate((padding,xEmb),axis=0)
  X[i] = xEmb
X_audio = np.array(X)

for i in range(len(X)):
  for j in range(len(X[i])):
    for k in range(len(X[i][j])):
      if np.isnan(X[i][j][k]):
        X_audio[i][j][k] = 0

pos = np.zeros(X_audio.shape)
for i in tqdm(range(len(X_audio))):
  for j in range(len(X_audio[i])):
    for d in range(len(X_audio[i][j])):
      if d %2 == 0:
        p = math.sin(j/pow(10000,d/62))
        pos[i][j][d] = p
      else:
        p = math.cos(j/pow(10000,(d-1)/62))
        pos[i][j][d] = p

print(X_text.shape,X_audio.shape,Xspeak.shape,pos.shape)

trainIndex = pd.read_csv("Train_index.csv")
trainIndex = trainIndex['index']
testIndex = pd.read_csv("Test_index.csv")
testIndex = testIndex['index']

X_text_Train = X_text[trainIndex]
X_text_Test = X_text[testIndex]
X_audio_Train = X_audio[trainIndex]
X_audio_Test = X_audio[testIndex]
X_pos_Train = pos[trainIndex]
X_pos_Test = pos[testIndex]
X_speak_Train = Xspeak[trainIndex]
X_speak_Test = Xspeak[testIndex]

YVals = pd.read_csv("Y_Volatility.csv")
YT3 = YVals['vFuture3']
YT7 = YVals['vFuture7']
YT15 = YVals['vFuture15']

Ys = [YT3,YT7,YT15]
YPrint = ["Tau 3","Tau 7","Tau 15"]

for i in range(3):
  YTrain = Ys[i][trainIndex]
  YTest = Ys[i][testIndex]
  
  modelN = "ModelV "+YPrint[i]+".h5"

  mc = tf.keras.callbacks.ModelCheckpoint(modelN, monitor='val_loss', verbose=0, save_best_only=True)
  model = createModelV(768, 62, 3, volatility_feedforward_size, volatility_hidden_dim, volatility_dropout, maxLen,maxSpeaker)
  model.compile(loss='mean_squared_error', optimizer=Adam(lr = learning_rate))
  out = model.fit([X_text_Train,X_audio_Train,X_pos_Train,X_speak_Train], YTrain, batch_size=batch_size, epochs=500, validation_data=([X_text_Test,X_audio_Test,X_pos_Test,X_speak_Test],YTest), verbose=0, callbacks=[mc])
  depen = {'MultiHeadSelfAttention': MultiHeadSelfAttention,'TransformerBlock': TransformerBlock} 
  model = load_model(modelN, custom_objects=depen)
  
  predTest = model.predict([X_text_Train,X_audio_Train,X_pos_Train,X_speak_Train])
  r = mean_squared_error(YTrain,predTest)
  print('MSE for Training Set for ',YPrint[i],': ',r)
  
  predTest = model.predict([X_text_Test,X_audio_Test,X_pos_Test,X_speak_Test])
  r = mean_squared_error(YTest,predTest)
  print('MSE for Testing Set for ',YPrint[i],': ',r)
  print()

YVals = pd.read_csv("Y_Movement.csv")
YT3 = YVals['YT3']
YT7 = YVals['YT7']
YT15 = YVals['YT15']

Ys = [YT3,YT7,YT15]
YPrint = ["Tau 3","Tau 7","Tau 15"]

for i in range(3):
  YTrain = Ys[i][trainIndex]
  YTest = Ys[i][testIndex]

  modelN = "ModelC "+YPrint[i]+".h5"

  mc = tf.keras.callbacks.ModelCheckpoint(modelN, monitor='val_accuracy', verbose=0, save_best_only=True)
  model = createModelC(768, 62, 3, movement_feedforward_size, movement_hidden_dim, movement_dropout, maxLen,maxSpeaker)
  model.compile(loss='binary_crossentropy', optimizer=Adam(lr = learning_rate), metrics=['accuracy'])
  out = model.fit([X_text_Train,X_audio_Train,X_pos_Train,X_speak_Train], YTrain, batch_size=batch_size, epochs=500, validation_data=([X_text_Test,X_audio_Test,X_pos_Test,X_speak_Test],YTest), verbose=0, callbacks=[mc])
  depen = {'MultiHeadSelfAttention': MultiHeadSelfAttention,'TransformerBlock': TransformerBlock} 
  model = load_model(modelN, custom_objects=depen)
  
  predTest = model.predict([X_text_Train,X_audio_Train,X_pos_Train,X_speak_Train]).round()
  mcc = matthews_corrcoef(YTrain, predTest)
  f1 = f1_score(YTrain,predTest)
  print('F1 for Training Set for ',YPrint[i],': ',f1)
  print('MCC for Training Set for ',YPrint[i],': ',mcc)  
  
  predTest = model.predict([X_text_Test,X_audio_Test,X_pos_Test,X_speak_Test]).round()
  mcc = matthews_corrcoef(YTest, predTest)
  f1 = f1_score(YTest,predTest)
  print('F1 for Testing Set for ',YPrint[i],': ',f1)
  print('MCC for Testing Set for ',YPrint[i],': ',mcc)
  print()