import time
import datetime
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pandas_datareader as pdr
from stockstats import*
import cv2
from PIL import Image

import keras
from keras.models import Sequential, Model
from keras.layers import *
from keras.layers import Conv2D, MaxPooling2D, MaxPooling3D
from keras import backend as K
from keras.callbacks import EarlyStopping

from sklearn.preprocessing import MinMaxScaler, StandardScaler
from keras.applications.xception import Xception
from tensorflow.keras.optimizers import Adam
import tensorflow as tf


TIME_RANGE = 40
PRICE_RANGE = 40
window_size = 60

"""Train"""

def scale_list(l, to_min, to_max):
    def scale_number(unscaled, to_min, to_max, from_min, from_max):
        return (to_max-to_min)*(unscaled-from_min)/(from_max-from_min)+to_min

    if len(set(l)) == 1:
        return [np.floor((to_max + to_min)/2)] * len(l)
    else:
        return [scale_number(i, to_min, to_max, min(l), max(l)) for i in l]

def getState(data, sell_option, t, TIME_RANGE, PRICE_RANGE):
    closing_values = data[0]
    macd = data[1]
    macds = data[2]
    #print(closing_values)
    half_scale_size = int(PRICE_RANGE / 2)

    graph_closing_values = list(np.round(scale_list(closing_values[t - TIME_RANGE:t], 0, half_scale_size - 1), 0))
    macd_data_together = list(
        np.round(scale_list(list(macd[t - TIME_RANGE:t]) + list(macds[t - TIME_RANGE:t]), 0, half_scale_size - 1), 0))
    graph_macd = macd_data_together[0:PRICE_RANGE]
    graph_macds = macd_data_together[PRICE_RANGE:]

    blank_matrix_macd = np.zeros((half_scale_size, TIME_RANGE, 3), dtype=np.uint8)
    x_ind = 0
    for s, d in zip(graph_macds, graph_macd):
        blank_matrix_macd[int(s), x_ind] = (0, 0, 255)
        blank_matrix_macd[int(d), x_ind] = (255, 175,0)
        x_ind += 1
    blank_matrix_macd = blank_matrix_macd[::-1]

    blank_matrix_close = np.zeros((half_scale_size, TIME_RANGE, 3), dtype=np.uint8)
    x_ind = 0
    if sell_option == 1:
      close_color = (0, 255, 0) #GREEN
    else:
      close_color = (255,0 , 0) #RED

    for v in graph_closing_values:
        blank_matrix_close[int(v), x_ind] = close_color
        x_ind += 1
    blank_matrix_close = blank_matrix_close[::-1]

    blank_matrix = np.vstack([blank_matrix_close, blank_matrix_macd])

    if 1 == 2:
        # graphed on matrix
        plt.imshow(blank_matrix)
        plt.show()

    return [blank_matrix]

def getStockData(key):
    stock_data = df = pdr.get_data_tiingo(key, api_key='9d4f4dacda5024f00eb8056b19009f32e58b38e5')

    stats = StockDataFrame.retype(stock_data)
    stock_data['Symbol'] = key

    stock_dif = (stock_data['close'] - stock_data['open'])
    stock_dif = stock_dif.values



    
    noise_ma_smoother = 1
    macd = stats.get('open_3_ema')
    #stats.get('close_{}_ema'.format(noise_ma_smoother))
    macd = macd.fillna(method='bfill')  
    macd = list(macd.values)

    longer_ma_smoother = 7
    macds = stats.get('open_15_ema')
    #stats.get('close_{}_ema'.format(longer_ma_smoother))
    macds = macds.fillna(method='bfill')  
    macds =  list(macds.values)

  
    closing_values = list(np.array(stock_data['close']))

    return_data = [closing_values, macd, macds]

    return return_data

def create_model(TIME_RANGE, PRICE_RANGE):
  
  input_shape_1 = (TIME_RANGE, PRICE_RANGE, 3)

  base_model = tf.keras.applications.resnet50.ResNet50( include_top=False, weights=None, input_shape=input_shape_1)

  action_size = 3

  x = base_model.output
  x = GlobalAveragePooling2D()(x)

  predictions = Dense(1, activation='linear')(x)

  model = Model(inputs = base_model.inputs, outputs = predictions)

  model.compile(loss='mse', optimizer='adam', metrics = ['accuracy'])
  return model

def create_model_2(TIME_RANGE, PRICE_RANGE):
  input_shape_1 = (60, TIME_RANGE, PRICE_RANGE, 3)

  model = Sequential()
  tf.keras.layers.ConvLSTM2D(64, (3, 3), input_shape=input_shape_1, return_sequences=False)
  model.add(Activation('relu'))
  model.add(MaxPooling2D(pool_size=(2, 2)))

  model.add(Conv2D(32, (2, 2)))
  model.add(Activation('relu'))
  model.add(MaxPooling2D(pool_size=(2, 2)))

  model.add(Flatten())  
  model.add(Dense(64))
  model.add(Activation('relu'))
  model.add(Dropout(0.5))
  model.add(Dense(1))
  model.add(Activation('sigmoid'))

  model.compile(loss='mse', optimizer='adam', metrics = ['accuracy'])

  return model

def load_model(TIME_RANGE, PRICE_RANGE):
    # use simple CNN structure
    in_shape = (window_size, TIME_RANGE, PRICE_RANGE, 3)
    model = Sequential()
    model.add(ConvLSTM2D(32, kernel_size=(7, 7), padding='valid', return_sequences=True, input_shape=in_shape))
    model.add(Activation('relu'))
    #model.add(MaxPooling3D(pool_size=(1, 2, 2)))
    model.add(ConvLSTM2D(64, kernel_size=(5, 5), padding='valid', return_sequences=True))
    model.add(MaxPooling3D(pool_size=(1, 2, 2)))
    model.add(ConvLSTM2D(96, kernel_size=(3, 3), padding='valid', return_sequences=True))
    model.add(Activation('relu'))
    model.add(ConvLSTM2D(96, kernel_size=(3, 3), padding='valid', return_sequences=True))
    model.add(Activation('relu'))
    model.add(ConvLSTM2D(96, kernel_size=(3, 3), padding='valid', return_sequences=True))
    model.add(MaxPooling3D(pool_size=(1, 2, 2)))
    model.add(Dense(320))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))

    out_shape = model.output_shape
    # print('====Model shape: ', out_shape)
    model.add(Reshape((window_size, out_shape[2] * out_shape[3] * out_shape[4])))
    model.add(LSTM(64, return_sequences=False))
    model.add(Dropout(0.5))
    model.add(Dense(1, activation='linear'))
    model.compile(loss='mse', optimizer='adam', metrics=['accuracy'])

    # model structure summary
    print(model.summary())

    return model

data = getStockData('PLUG')
train_data = []

for t in range(50, len(data[0])):
  train_data.append(getState(data, 0, t, PRICE_RANGE, TIME_RANGE))

train_data = np.array(train_data)

train_data.shape

train_data = np.reshape(train_data, (train_data.shape[0], train_data.shape[2], train_data.shape[3], train_data.shape[4]))
train_data.shape

close_vals = np.array(data[0])

close_vals.shape

y_train = np.round((close_vals[1 : len(data[0])] -  close_vals[0 : len(data[0])- 1]), 3)

x_train = []
for i in range(window_size, len(data[0]) - window_size - 1):
  x_train.append(train_data[i - window_size: i ])
x_train = np.array(x_train)

y_train = y_train[0:x_train.shape[0]]

y_train = np.array(y_train)

for i in range(len(x_train)):
  print(x_train[i].shape)

img_rows, img_cols = TIME_RANGE, PRICE_RANGE


x_train = x_train.astype('float32')
x_train.shape

model = load_model(TIME_RANGE, PRICE_RANGE)

tf.keras.utils.plot_model(model, 'mulit_input_output_model.png', show_shapes=True)

model.fit(x_train, y_train, batch_size=512, epochs=10)

model.save('')