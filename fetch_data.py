import datetime
import numpy as np
import pandas as pd
import pandas_datareader.data as web
import yfinance as yf
yf.pdr_override()


def generateData_SP500():
    start = datetime.datetime(2018, 1, 4)
    end = datetime.datetime(2019, 1, 1)
    table = pd.read_html('https://en.wikipedia.org/wiki/List_of_S%26P_500_companies')
    splist = table[0].Symbol.values
    temp = web.get_data_yahoo(splist[0], start, end)
    df = pd.DataFrame(index=temp.index)
    df_volume=pd.DataFrame(index=temp.index)
    df[splist[0]] = temp['Adj Close']
    df_volume[splist[0]]=temp['Volume']
    for ticker in splist[1:]:
        temp = web.get_data_yahoo(ticker, start, end)
        df[ticker] = temp['Adj Close']
        df_volume[ticker]=temp['Volume']

    return (df,df_volume)

df,df_volume=generateData_SP500()
