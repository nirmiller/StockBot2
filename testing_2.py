import keras
from keras.models import load_model
import os
from handling_2 import *

stock_name, model_name = "PLUG", "/content/drive/MyDrive/StockBot/models/stock_bot_comp/CNN/model_3/model_3_4_20"
model = load_model(model_name)
window_size = model.layers[0].input.shape.as_list()[1]

agent = Agent(TIME_RANGE, PRICE_RANGE, True, model_name=model_name)
data = getStockData(stock_name)

l = len(data[0])
agent.total_inventory.append(0)
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

    if action == 0 and equity - (buy * close) > 0:  # buy
        equity -= buy * close
        agent.inventory += buy
        sell_option = 1
        print("Buy: {} Amount : {}".format(close, buy))
    elif action == 1 and agent.inventory >= sell:  # sell
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
    elif action == 0:
        print("Hold")

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
