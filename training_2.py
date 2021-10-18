import matplotlib.pyplot as plt

plt.style.use('fivethirtyeight')
from handling_2 import *

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

'''
stocks = ['PLUG']
forecast_window = 60
f_bot = F_Bot(forecast_window)
forecast = f_bot.getForecastData(stocks)
# print(forecast)
'''

import sys
import math

np.seterr(divide='ignore', invalid='ignore')

memory_count = 0

stock_name, episode_count = "PLUG", 100
agent = Agent(TIME_RANGE, PRICE_RANGE)
data = getStockData(stock_name)
close_values = data[0]

l = len(close_values)
agent.total_inventory.append(0)

for e in range(episode_count + 1):
    print("Episode " + str(e) + "/" + str(episode_count))
    total_profit = 0
    initial_profit = 0
    agent.inventory = 0
    batch_size = 32
    equity = 100_000
    initial_equity = 0
    change_equity = 0
    buy = 1
    sell = 1
    sell_option = 0

    profit_data = [[0], [equity]]
    state = getState(data, sell_option, TIME_RANGE, TIME_RANGE, PRICE_RANGE)
    count = 1

    for t in range(TIME_RANGE, l):

        action = agent.act(state)

        # sit
        next_state = getState(data, sell_option, t + 1, TIME_RANGE, PRICE_RANGE)
        # print(next_state)
        reward = 0
        close = data[0][t]

        buy = math.floor(equity / close)
        sell = agent.inventory
        # print("Close = {}  :   Open = {}  :  Volume = {}".format(close, data[1][t], data[2][t]))

        print("Close = {} Money = {},  Inventory = {}".format(close, profit_data[0][-1], agent.total_inventory[-1]))

        if t == TIME_RANGE:
            initial_equity = equity

        if action == 0:  # buy
            equity -= buy * close
            agent.inventory += buy
            sell_option = 1
            print("Buy: {} Amount : {}".format(close, buy))
        elif action == 1 and agent.inventory >= sell :  # sell
            equity += sell * close
            change_equity = equity - initial_equity
            initial_profit = total_profit
            total_profit += change_equity
            if change_equity < 0:
                reward = change_equity
            else:
                reward = ((total_profit + equity) / (initial_equity)) * change_equity

            agent.inventory -= sell
            initial_equity = equity
            sell_option = 0
            print("Sell: {} Amount : {} | Profit: {} | Equity : {}".format(close, sell, change_equity, equity))
            count = 0

        print(f"Reward : {reward}")
        agent.total_inventory.append(agent.inventory)

        profit_data[0].append(total_profit)
        profit_data[1].append(initial_equity)

        done = True if t == l - 1 else False
        agent.memory.append((state, action, reward, next_state, done))
        state = next_state

        if done:
            equity += agent.inventory * close
            print("--------------------------------")
            print("Total Profit:" + formatPrice(equity))
            print(e)
            print("--------------------------------")

        if len(agent.memory) > batch_size:
            agent.expReplay(batch_size)
            print("REPLAY {}".format(agent.epsilon))

    if e % 10 == 0:
        agent.model.save("/content/drive/MyDrive/StockBot/models/stock_bot_comp/CNN/model_3/model_3_2_{}".format(str(e)))

    if e % 7 == 0:
        agent.epsilon = 0.5
