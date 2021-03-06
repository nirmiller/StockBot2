import matplotlib.pyplot as plt

plt.style.use('fivethirtyeight')

import time
import datetime
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pandas_datareader as pdr
from stockstats import *
import cv2
from PIL import Image

import keras
from keras.models import Sequential, Model
from keras.layers import Dense, Dropout, Flatten, GlobalAveragePooling2D
from keras.layers import Conv2D, MaxPooling2D
from keras import backend as K
from keras.callbacks import EarlyStopping

from sklearn.preprocessing import MinMaxScaler, StandardScaler, Normalizer, RobustScaler, MaxAbsScaler, PowerTransformer
from keras.applications.xception import Xception
from tensorflow.keras.optimizers import Adam
import tensorflow as tf
import math

TIME_RANGE, PRICE_RANGE = 40, 40


def scale_list(l, to_min, to_max):
    def scale_number(unscaled, to_min, to_max, from_min, from_max):
        return (to_max - to_min) * (unscaled - from_min) / (from_max - from_min) + to_min

    if len(set(l)) == 1:
        return [np.floor((to_max + to_min) / 2)] * len(l)
    else:
        return [scale_number(i, to_min, to_max, min(l), max(l)) for i in l]


def getState(data, sell_option, t, TIME_RANGE, PRICE_RANGE):
    closing_values = data[0]
    macd = data[1]
    macds = data[2]
    # print(closing_values)
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
        blank_matrix_macd[int(d), x_ind] = (255, 175, 0)
        x_ind += 1
    blank_matrix_macd = blank_matrix_macd[::-1]

    blank_matrix_close = np.zeros((half_scale_size, TIME_RANGE, 3), dtype=np.uint8)
    x_ind = 0
    if sell_option == 1:
        close_color = (0, 255, 0)  # GREEN
    else:
        close_color = (255, 0, 0)  # RED

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
    stock_data = pdr.get_data_tiingo(key, start='8-14-2020', api_key='9d4f4dacda5024f00eb8056b19009f32e58b38e5')
    stock_data = np.array(stock_data)

    stats = StockDataFrame.retype(stock_data)
    stock_data['Symbol'] = key

    stock_dif = (stock_data['close'] - stock_data['open'])
    stock_dif = stock_dif.values

    noise_ma_smoother = 1
    macd = stats.get('macd')
    # stats.get('close_{}_ema'.format(noise_ma_smoother))
    macd = macd.fillna(method='bfill')
    macd = list(macd.values)

    longer_ma_smoother = 7
    macds = stats.get('macds')
    # stats.get('close_{}_ema'.format(longer_ma_smoother))
    macds = macds.fillna(method='bfill')
    macds = list(macds.values)

    closing_values = list(np.array(stock_data['close']))

    return_data = [closing_values, macd, macds]

    return return_data


# prints formatted price
def formatPrice(n):
    return ("-$" if n < 0 else "$") + "{0:.2f}".format(abs(n))


def getBotPeformance(raw_data, window_size):
    # DATA is peformance data of bot, so : Total Profit made by bot in percent
    # DATA = total_profit |  initial_profit *should only change if bot is selling
    raw_data = np.array(raw_data)
    data = (raw_data[0]) / (abs(raw_data[1]))
    # ADD more transformations later
    for i in range(data.shape[0]):
        if math.isnan(data[i]):
            data[i] = 0
    # data[i] = sigmoid(data[i])

    return_data = []
    if data.shape[0] >= window_size:
        return_data = data[-window_size:]
    else:
        d = (window_size - data.shape[0] + 1)
        return_data = np.concatenate((data[0:-1], d * [data[-1]]), axis=None)
    return return_data


def fix_input(state):
    state = np.array(state)
    img_rows, img_cols = TIME_RANGE, PRICE_RANGE
    state = np.reshape(state, (state.shape[0], img_rows, img_cols, 3))
    state = state.astype('float32')
    return state


import keras
from keras.models import Sequential
from keras.models import load_model
from keras.layers import Dense, LSTM, Dropout
from tensorflow.keras.optimizers import Adam
import tensorflow as tf


class F_Bot:
    def __init__(self, window_size, model_name=""):
        self.model_name = model_name
        self.window_size = window_size
        self.forecast_model = self._model(self.window_size)

    def create_model(self):
        input_shape_1 = (self.time_range, self.price_range, 3)

        model = Sequential()
        model.add(Conv2D(32, kernel_size=(5, 5), activation='relu', input_shape=input_shape_1))
        model.add(Conv2D(64, kernel_size=(4, 4), activation='relu'))
        model.add(Conv2D(64, kernel_size=(3, 3), activation='relu'))
        model.add(Flatten())
        model.add(Dense(512))
        model.add(Dense(self.action_size, activation='linear'))

        model.compile(loss='mse', optimizer=Adam(learning_rate=.001), metrics=['accuracy'])

        return model

    def getForecastData(self, stocks):
        forecast = []
        window_size = self.window_size
        # Use F-Bot to calculate the n* forecasting values for the n* # of stocks
        number_stocks = len(stocks)
        for i in range(0, number_stocks):
            stock = stocks[i]
            df = pdr.get_data_tiingo(stock, start='8-14-2020', api_key='9d4f4dacda5024f00eb8056b19009f32e58b38e5')
            data = df.filter(['close', 'open', 'high', 'low'])
            data = data[30:-1]
            dataset = data.values
            normalizer = Normalizer()
            scaler = RobustScaler()
            standard = MinMaxScaler(feature_range=(0, 1))
            # normalizer = Normalizer()
            scaled_data = normalizer.fit_transform(dataset)
            scaled_data = scaler.fit_transform(scaled_data)
            scaled_data = standard.fit_transform(scaled_data)

            # VOLUME DATA

            v_data = df.filter(['volume'])
            v_data = v_data[30:-1].values
            volume_data = []
            for i in range(len(dataset) - 1):
                volume_data.append(v_data[i + 1] - v_data[i])

            s = MinMaxScaler(feature_range=(-1, 1))

            volume_data = np.array(volume_data)
            volume_data = np.reshape(volume_data, (volume_data.shape[0], 1))

            scaled_volume = s.fit_transform(volume_data)

            scaled_volume = np.reshape(scaled_volume, (1, scaled_volume.shape[0]))

            volume_vals = scaled_volume[0].tolist()

            volume_vals = np.array(volume_vals)

            # ------------------------------------------------------------#

            train_data = scaled_data[0:-1, :]

            x_vals = [[], [], [], [], []]

            for i in range(window_size, len(train_data) - 1):
                for j in range(0, train_data.shape[1] + 1):
                    if j <= train_data.shape[1] - 1:
                        x_vals[j].append(train_data[i - window_size: i, j])
                    else:
                        x_vals[j].append(volume_vals[i - window_size: i])
            x_vals = np.array(x_vals)

            x_vals = np.reshape(x_vals, (x_vals.shape[1], x_vals.shape[2], x_vals.shape[0]))
            predictions = self.forecast_model.predict(x_vals)
            predictions = np.array(predictions)
            predictions = np.reshape(predictions, (predictions.shape[1], predictions.shape[0]))
            # print(predictions[0])
            forecast.append(predictions[0])
        return forecast


import keras
from keras.models import Sequential
from keras.models import load_model
from keras.layers import Dense, LSTM, Dropout
from tensorflow.keras.optimizers import Adam
import tensorflow as tf

import numpy as np
import random
from collections import deque


class Agent:
    def __init__(self, PRICE_RANGE, TIME_RANGE, is_eval=False, model_name=""):
        self.price_range = PRICE_RANGE
        self.time_range = TIME_RANGE

        self.action_size = 3  # sit, buy, sell
        self.memory = deque(maxlen=500_000)
        self.inventory = 0
        self.model_name = model_name
        self.is_eval = is_eval
        self.total_inventory = []

        self.gamma = 0.92
        self.epsilon = 0.8
        self.epsilon_min = 0.001
        self.epsilon_decay = 0.9988

        if is_eval:
            self.model = load_model(model_name)
        else:
            self.model = self.create_model()

        # self.model = load_model('/content/drive/MyDrive/StockBot/models/stock_bot_comp/CNN/model_4/model_4_1_20')

    def create_model(self):
        input_shape_1 = (self.time_range, self.price_range, 3)

        model = Sequential()
        model.add(Conv2D(32, kernel_size=(5, 5), activation='relu', input_shape=input_shape_1))
        model.add(Conv2D(64, kernel_size=(4, 4), activation='relu'))
        model.add(Conv2D(64, kernel_size=(3, 3), activation='relu'))
        model.add(Flatten())
        model.add(Dense(256))
        model.add(Dense(self.action_size, activation='linear'))

        model.compile(loss='mse', optimizer=Adam(learning_rate=.001), metrics=['accuracy'])

        return model

    def act(self, state):
        if not self.is_eval and random.random() <= self.epsilon:
            return random.randrange(self.action_size)
        options = self.model.predict(fix_input(state))
        print(options)
        return np.argmax(options[0])

    def expReplay(self, batch_size):
        mini_batch = []
        l = len(self.memory)
        for i in range(l - batch_size + 1, l):
            mini_batch.append(self.memory[i])
        #	mini_batch = random.sample(self.memory, batch_size)

        for state, action, reward, next_state, done in mini_batch:
            target = reward
            state = fix_input(state)
            if not done:
                target = reward + self.gamma * np.amax(self.model.predict(fix_input(next_state))[0])

            target_f = self.model.predict(state)
            target_f[0][action] = target

            self.model.fit(state, target_f, epochs=1, verbose=0)
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
