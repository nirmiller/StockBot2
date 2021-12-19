import pandas_datareader as pdr
import numpy as np

stock_data = pdr.get_data_tiingo('PLUG', start='8-14-2020', api_key='9d4f4dacda5024f00eb8056b19009f32e58b38e5')
print(stock_data)