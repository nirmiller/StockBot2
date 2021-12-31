'''
#Take model and minute stock data
-Augment minute stock data with optimal actions taken
-Train model through best actions to take during these times
'''


import matplotlib.pyplot as plt

plt.style.use('fivethirtyeight')
from minute_handling import *

data = getStockData("NaN")

x_data = []
l = len(data[0])

for i in range(TIME_RANGE, l):
    snap = getState(data, 1, i, TIME_RANGE, PRICE_RANGE)
    x_data.append(snap)

x_data = np.array(fix_input(x_data))

