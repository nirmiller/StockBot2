import keras
from keras.models import load_model
import os
from handling import *

stock_name = "PLUG"
model = load_model("CNN/model_1/model_1_1_0")
window_size = model.layers[0].input.shape.as_list()[1]

data = getStockData(stock_name)
max_profit = 0
max_i = 0
total_data = []

for i in range(4, 11):
    model_name = "CNN/model_1/model_1_1_{}".format(
        2 * i)
    agent = Agent(TIME_RANGE, PRICE_RANGE, True, model_name)
    l = len(data[0]) - 1
    agent.total_inventory.append(1)
    total_profit = 0
    initial_profit = 0
    agent.inventory = 0
    batch_size = 32
    equity = 10_000
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

        if (action == 2 and agent.inventory == 0) or (action == 1 and equity - (buy * close) <= 0) or (
                action == 1 and buy <= 0):
            print("Hold due to circumstances {}".format(action))
            reward = -2500
        elif action == 1 and equity - (buy * close) > 0:  # buy
            equity -= buy * close
            agent.inventory += buy
            sell_option = 1
            print("Buy: {} Amount : {}".format(close, buy))
            reward = 1000
        elif action == 2 and agent.inventory > 0:  # sell
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

        if equity > max_profit:
            max_profit = equity
            max_i = i * 2

print(max_profit)
print(max_i)
