# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.
import numpy as np
import pandas as pd
import warnings
from sklearn.linear_model import LinearRegression
import scipy.cluster.hierarchy as sch


class backtest_model:
    """
    Given a user-defined portfolio construction strategy (a function that takes in stock-related data and returns portfolio weights) and
    the data that the user wish the strategy to be tested on, calculate several evaluation metrics of the portfolio, including
    net_returns, sharpe ratio, certainty equivalent returns, turnover, etc.
    Various inputs can be modified to suit the needs of strategy and backtesting scenarios, such as price-impact models,
    transaction costs, etc.
    """

    def __init__(self, strategy, involved_data_type, need_extra_data=False, trace_back=False, name='Unnamed', missing_val=False):
        """
        Initiate the model with the strategy function, and clarify involved data types needed, whose sequence MUST be consistent
        with that of the list of dataframes used inside strategy function

        :param strategy: user-defined function that serves as portfolio construction strategy
        :type strategy: function

        :param involved_data_type: a list of strings that indicate the type of data {'price','return','ex_return'} used in the strategy, the order of the strings will be the order that data are passed to the strategy
        :type involved_data_type: list

        :param need_extra_data: indicate whether the strategy need extra_data (data other than {'price','return','ex_return'}) to function. Note: 1. the datetime index of extra_data must match that of the provided data. 2. change-of-frequency functionality will be suspended if extra data is needed
        :type need_extra_data: bool

        :param trace_back: indicate whether the strategy need to trace back to past portfolios to function. Note: please handle the boundary situation where past portfolios is empty in the strategy function
        :type trace_back: bool

        :param name: name of the strategy to be tested
        :type name: str

        :param missing_val : indicate whether user strategy function can handle missing values in the data on its own. True means the function can deal with missing values. False means it cannot. A wrapper function would be applied to the strategy function to deal with missing data.
        :type missing_val: bool
        """

        def wrapper(function, list_df):
            df = list_df[0]
            position_nan = df.isna().any().values
            w = np.zeros(df.shape[1])
            w[position_nan == False] = function([df.dropna(axis=1)])
            return w

        if not missing_val:
            self.__strategy = lambda x: wrapper(strategy, x)
        else:
            self.__strategy = strategy

        if type(involved_data_type) != list:
            raise Exception('"involved_data_type" must be given in a list')
        else:
            self.__involved_data_type = involved_data_type

        if type(need_extra_data) != bool:
            raise Exception('"need_extra_data" must be a bool variable')
        else:
            self.__need_extra_data = need_extra_data

        if type(trace_back) != bool:
            raise Exception('"trace_back" must be a bool variable')
        else:
            self.__trace_back = trace_back

        if type(name) != str:
            raise Exception('"name" must be a string variable')
        else:
            self.__name = name

        self.__last_test_frequency = None
        self.__last_test_portfolios = None
        self.__price_impact = False
        self.__sharpe = None
        self.__ceq = None
        self.__average_turnover = None
        self.__total_turnover = None
        self.__net_returns = None
        self.__net_excess_returns = None

    # function to prepare data, including change of frequency, convert between price, return and ex_return
    def __prepare_data(self, data, freq_data, data_type, rf, interval, window, freq_strategy,
                       volume=pd.DataFrame(), price_impact=False):

        if not isinstance(data, pd.DataFrame):
            raise Exception('Please provide correct format of test data!')

        try:
            data.index = pd.to_datetime(data.index)
        except:
            print(
                'Invalid index provided in your test data, please make sure that index is in compatible datetime format')

        volume.index = pd.to_datetime(volume.index)
        data = data.copy()

        if data_type == 'return':
            if freq_data != freq_strategy:
                warnings.warn(
                    'data_type==return with interval>1 or change of frequency, Expect large amount of computational error')
                data['###rf'] = rf  # add 'rf' to the dataframe to go through transformation together
                data = (1 + data).apply(lambda x: np.cumprod(x))
                data = data.resample(freq_strategy).ffill().pct_change().dropna(axis=0, how='all')
                normal_return_df = data.iloc[:,:-1]
                risk_free_df=data.iloc[:,-1]
                excess_return_df = normal_return_df.sub(risk_free_df.values, axis=0).dropna(axis=0, how='all')
                return (normal_return_df, excess_return_df, risk_free_df,
                        pd.DataFrame(index=normal_return_df.index))
            else:
                normal_return_df = data
                excess_return_df = normal_return_df.sub(rf.values, axis=0)
                return (normal_return_df, excess_return_df, rf.loc[normal_return_df.index],
                        pd.DataFrame(index=normal_return_df.index))
        elif data_type == 'ex_return':
            if freq_data != freq_strategy:
                warnings.warn(
                    'data_type==ex_return with interval>1 or change of frequency, Expect large amount of computational error')
                data = data.add(rf, axis=0)
                data['###rf'] = rf  # add 'rf' to the dataframe to go through transformation together
                data = (1 + data).apply(lambda x: np.cumprod(x))
                data = data.resample(freq_strategy).ffill().pct_change().dropna(axis=0, how='all')
                normal_return_df = data.iloc[:, :-1]
                risk_free_df = data.iloc[:, -1]
                excess_return_df = normal_return_df.sub(risk_free_df.values, axis=0).dropna(axis=0, how='all')
                return (normal_return_df, excess_return_df, risk_free_df,
                        pd.DataFrame(index=normal_return_df.index))
            else:
                excess_return_df = data
                normal_return_df = excess_return_df.add(rf, axis=0)
                return (normal_return_df, excess_return_df, rf.loc[normal_return_df.index],
                        pd.DataFrame(index=normal_return_df.index))
        elif data_type == 'price':
            #data['###rf'] = rf  # add 'rf' to the dataframe to go through transformation together
            rf_df=np.cumprod(1+rf)
            if freq_data != freq_strategy:
                data = data.resample(freq_strategy).ffill()
                rf_df=rf_df.resample(freq_strategy).ffill()
                if price_impact:
                    volume = volume.resample(freq_strategy).mean()
            normal_return_df = data.pct_change().dropna(axis=0, how='all')
            risk_free_df=rf_df.pct_change().dropna(axis=0,how='all')
            excess_return_df = normal_return_df.sub(risk_free_df.values, axis=0)
            if price_impact:
                return (normal_return_df, excess_return_df, volume.loc[normal_return_df.index],
                        risk_free_df,
                        data.loc[normal_return_df.index])
            else:
                return (normal_return_df, excess_return_df, risk_free_df,
                        data.loc[normal_return_df.index])

    # rebalance function to be applied to each rolling window of length (window)
    def __rebalance(self, ex_return_df, normal_return_df, price_df, window, extra_data=None):
        historical_portfolios = []
        map = {'price': price_df, 'ex_return': ex_return_df, 'return': normal_return_df}
        if self.__need_extra_data:
            if self.__trace_back:
                for df in ex_return_df.rolling(window):
                    if df.shape[0] >= window:
                        historical_portfolios.append(
                            self.__strategy([map[i].loc[df.index] for i in self.__involved_data_type],
                                            extra_data.loc[df.index],
                                            historical_portfolios))
            else:
                for df in ex_return_df.rolling(window):
                    if df.shape[0] >= window:
                        historical_portfolios.append(
                            self.__strategy([map[i].loc[df.index] for i in self.__involved_data_type],
                                            extra_data.loc[df.index]))
        else:
            if self.__trace_back:
                for df in ex_return_df.rolling(window):
                    if df.shape[0] >= window:
                        historical_portfolios.append(
                            self.__strategy([map[i].loc[df.index] for i in self.__involved_data_type],
                                            historical_portfolios))
            else:
                for df in ex_return_df.rolling(window):
                    if df.shape[0] >= window:
                        historical_portfolios.append(
                            self.__strategy([map[i].loc[df.index] for i in self.__involved_data_type]))
        return historical_portfolios

    def __test_price_impact(self, data, freq_data, data_type, rf, interval, window, freq_strategy, ptc_buy,
                            ptc_sell, ftc, volume, c, initial_wealth, extra_data, price_impact_model='default'):
        # prepare data
        normal_return_df, excess_return_df, volume, risk_free_rate, price_df = self.__prepare_data(data, freq_data,
                                                                                                   data_type, rf,
                                                                                                   interval, window,
                                                                                                   freq_strategy,
                                                                                                   volume,
                                                                                                   price_impact=True)

        T = excess_return_df.shape[0]  # length of dataset
        N = excess_return_df.shape[1]  # number of assets
        if window < N:
            warnings.warn('window length smaller than the number of assets, may not get feasible portfolios')
        if window >= T - 2:  # 2 here can change later
            raise Exception(
                'Too few samples to test on will result in poor performance : reduce window or decrease interval or '
                'increase length of data')

        # apply rolling windows with __rebalance
        portfolios = self.__rebalance(excess_return_df, normal_return_df, price_df, window, extra_data)

        try:
            assert sum(portfolios[0]) <= 1 + 0.000001
        except:
            raise Exception(
                'Please make sure your strategy builds a portfolios whose sum of weights does not exceed 1!')

        portfolios = pd.DataFrame(portfolios).iloc[::interval]

        # save the portfolios for calling
        self.__last_test_portfolios = portfolios.set_axis(excess_return_df.columns.values, axis='columns').set_axis(
            excess_return_df.iloc[window - 1::interval].index.values, axis='index')

        if interval > 1:
            if price_df.empty:
                df=normal_return_df.join(risk_free_rate)
                df=(1+df.iloc[window-1:]).apply(lambda x:np.cumprod(x)).iloc[::interval].pct_change().dropna(axis=0,how='all')
                normal_return_df=df.iloc[:,:-1]
                risk_free_rate=df.iloc[:,-1]
                excess_return_df = normal_return_df.sub(risk_free_rate.values, axis=0)
                price_df = price_df.iloc[window - 1::interval].iloc[1:]
            else:
                price_df = price_df.iloc[window - 1::interval]
                normal_return_df=price_df.pct_change().dropna(axis=0,how='all')
                risk_free_rate=np.cumprod(1+risk_free_rate[window-1:]).iloc[::interval].pct_change().dropna(axis=0,how='all')
                excess_return_df=normal_return_df.sub(risk_free_rate.values, axis=0)
                price_df=price_df.iloc[1:]
        else:
            excess_return_df = excess_return_df.iloc[window:]
            normal_return_df = normal_return_df.iloc[window:]
            risk_free_rate = risk_free_rate.iloc[window:]
            price_df = price_df.iloc[window:]

        # pre_balance portfolios that serves as denominators
        pre_balance_portfolios = (1 + normal_return_df).mul(portfolios.iloc[:-1].values)

        # turnover
        # normalise portfolio weight before rebalancing at the start of each period
        # note that turnover ratio is not affected by price-impact model
        pre_balance_portfolios = pre_balance_portfolios.div(pre_balance_portfolios.sum(axis=1).values, axis=0)
        diff = (portfolios.iloc[1:].sub(pre_balance_portfolios.values)).dropna(axis=0, how='all')
        self.__total_turnover = abs(diff).sum(axis=1).sum()
        self.__average_turnover = self.__total_turnover / (T - window)

        # pre_balance portfolios that serves as nominators
        pre_balance_portfolios_2 = (1 + normal_return_df.iloc[1:]).mul(portfolios.iloc[1:-1].values)

        # factor in the initial_wealth for all 'diff','portfolios'
        portfolios *= initial_wealth
        pre_balance_portfolios *= initial_wealth
        pre_balance_portfolios_2 *= initial_wealth
        diff *= initial_wealth

        # transform volume to average volume
        volume = volume.rolling(window).mean().dropna(axis=0, how='all').loc[normal_return_df.index]

        # evolution of money account
        pre_balance_money = np.zeros(risk_free_rate.shape[0])

        # Money account value after each period, before rebalancing

        pi_models = {'default': {'buy': 1 + c * (diff[diff >= 0].div((volume * price_df).values)) ** 0.6,
                                 'sell': 1 - c * (abs(diff[diff < 0]).div((volume * price_df).values)) ** 0.6}}
        pi_buy, pi_sell = pi_models[price_impact_model]['buy'], pi_models[price_impact_model]['sell']

        # sell = ((abs(diff[diff < 0]).mul(1 - ptc_sell)) * (
        #         1 - c * (abs(diff[diff < 0]).div((volume * price_df).values)) ** 0.6)).sum(axis=1)
        # buy = ((diff[diff >= 0].mul(1 + ptc_buy)) * (
        #         1 + c * (diff[diff >= 0].div((volume * price_df).values)) ** 0.6)).sum(axis=1)
        sell = ((abs(diff[diff < 0]).mul(1 - ptc_sell)) * pi_sell).sum(axis=1)
        buy = ((diff[diff >= 0].mul(1 + ptc_buy)) * pi_buy).sum(axis=1)
        fixed = diff[diff != 0].count(axis=1).mul(ftc)
        after_balance_money = pre_balance_money + sell - buy - fixed
        pre_balance_money_2 = after_balance_money[:-1].mul((1 + risk_free_rate.iloc[1:]).values)

        # net_returns
        self.__net_returns = (pre_balance_portfolios_2.sum(axis=1).add(pre_balance_money_2.values)).div(
            pre_balance_portfolios.sum(axis=1).add(pre_balance_money).iloc[:-1].values) - 1

        self.__net_excess_returns = self.__net_returns.sub(risk_free_rate.iloc[1:].values)

        self.__sharpe = np.mean(self.__net_excess_returns) / np.std(self.__net_excess_returns, ddof=1)

    def __test_no_price_impact(self, data, freq_data, data_type, rf, interval, window, freq_strategy, ptc_buy,
                               ptc_sell, ftc, initial_wealth, extra_data):
        # prepare data
        normal_return_df, excess_return_df, risk_free_rate, price_df = self.__prepare_data(data, freq_data,
                                                                                           data_type, rf,
                                                                                           interval, window,
                                                                                           freq_strategy)

        T = excess_return_df.shape[0]  # length of dataset
        N = excess_return_df.shape[1]  # number of assets
        if window < N:
            warnings.warn('window length smaller than the number of assets, may not get feasible portfolios')
        if window >= T - 2:  # 3 here can change later
            raise Exception(
                'Too few samples to test on will result in poor performance : reduce window or decrease interval or '
                'increase length of data')

        # apply rolling windows with __rebalance
        portfolios = self.__rebalance(excess_return_df, normal_return_df, price_df, window, extra_data)

        try:
            assert sum(portfolios[0]) <= 1 + 0.000001
        except:
            raise Exception(
                'Please make sure your strategy builds a portfolios whose sum of weights does not exceed 1!')

        portfolios = pd.DataFrame(portfolios).iloc[::interval]

        # save the portfolios for calling
        self.__last_test_portfolios = portfolios.set_axis(excess_return_df.columns.values, axis='columns').set_axis(
            excess_return_df.iloc[window - 1::interval].index.values, axis='index')

        if interval > 1:
            if price_df.empty:
                df = normal_return_df.join(risk_free_rate)
                df = (1 + df.iloc[window - 1:]).apply(lambda x: np.cumprod(x)).iloc[::interval].pct_change().dropna(
                    axis=0, how='all')
                normal_return_df = df.iloc[:, :-1]
                risk_free_rate = df.iloc[:, -1]
                excess_return_df = normal_return_df.sub(risk_free_rate.values, axis=0)
                price_df = price_df.iloc[window - 1::interval].iloc[1:]
            else:
                price_df = price_df.iloc[window - 1::interval]
                normal_return_df = price_df.pct_change().dropna(axis=0, how='all')
                risk_free_rate=np.cumprod(1+risk_free_rate[window-1:]).iloc[::interval].pct_change().dropna(axis=0,how='all')
                excess_return_df = normal_return_df.sub(risk_free_rate.values, axis=0)
                price_df = price_df.iloc[1:]
        else:
            excess_return_df = excess_return_df.iloc[window:]
            normal_return_df = normal_return_df.iloc[window:]
            risk_free_rate = risk_free_rate.iloc[window:]
            price_df = price_df.iloc[window:]

        # pre_balance portfolios that serves as denominators
        pre_balance_portfolios = (1 + normal_return_df).mul(portfolios.iloc[:-1].values)

        # turnover
        # normalise portfolio weight before rebalancing at the start of each period
        # note that turnover ratio is not affected by price-impact model
        pre_balance_portfolios = pre_balance_portfolios.div(pre_balance_portfolios.sum(axis=1).values, axis=0)
        diff = (portfolios.iloc[1:].sub(pre_balance_portfolios.values)).dropna(axis=0, how='all')
        self.__total_turnover = abs(diff).sum(axis=1).sum()
        self.__average_turnover = self.__total_turnover / (T - window)

        # pre_balance portfolios that serves as nominators
        pre_balance_portfolios_2 = (1 + normal_return_df.iloc[1:]).mul(portfolios.iloc[1:-1].values)

        if ftc != 0:
            # factor in the initial_wealth for all 'diff','portfolios'
            portfolios *= initial_wealth
            pre_balance_portfolios *= initial_wealth
            pre_balance_portfolios_2 *= initial_wealth
            diff *= initial_wealth

            # transaction cost impacts
            if ptc_buy == ptc_sell == 0:
                excess_returns = excess_return_df.mul(portfolios.iloc[:-1].values).sum(axis=1)
                self.__net_excess_returns = excess_returns[1:]
                self.__net_returns = self.__net_excess_returns.add(risk_free_rate.iloc[1:].values)
                # self.__net_excess_returns = excess_returns
            else:
                sell = (abs(diff[diff < 0]).mul(1 - ptc_sell)).sum(axis=1)
                buy = (diff[diff >= 0].mul(1 + ptc_buy)).sum(axis=1)
                fixed = diff[diff != 0].count(axis=1).mul(ftc)
                # evolution of money account
                pre_balance_money = np.zeros(risk_free_rate.shape[0])
                after_balance_money = pre_balance_money + sell - buy - fixed
                pre_balance_money_2 = after_balance_money[:-1].mul((1 + risk_free_rate.iloc[1:]).values)

                self.__net_returns = (pre_balance_portfolios_2.sum(axis=1).add(pre_balance_money_2.values)).div(
                    pre_balance_portfolios.sum(axis=1).add(pre_balance_money).iloc[:-1].values) - 1

                self.__net_excess_returns = self.__net_returns.sub(risk_free_rate.iloc[1:].values)
        else:
            # transaction cost impacts
            if ptc_buy == ptc_sell == 0:
                excess_returns = excess_return_df.mul(portfolios.iloc[:-1].values).sum(axis=1)
                self.__net_excess_returns = excess_returns[1:]
                self.__net_returns = self.__net_excess_returns.add(risk_free_rate.iloc[1:].values)
                # self.__net_excess_returns = excess_returns
            else:
                sell = (abs(diff[diff < 0]).mul(1 - ptc_sell)).sum(axis=1)
                buy = (diff[diff >= 0].mul(1 + ptc_buy)).sum(axis=1)
                # evolution of money account
                pre_balance_money = np.zeros(risk_free_rate.shape[0])
                after_balance_money = pre_balance_money + sell - buy
                pre_balance_money_2 = after_balance_money[:-1].mul((1 + risk_free_rate.iloc[1:]).values)

                self.__net_returns = (pre_balance_portfolios_2.sum(axis=1).add(pre_balance_money_2.values)).div(
                    pre_balance_portfolios.sum(axis=1).add(pre_balance_money).iloc[:-1].values) - 1

                self.__net_excess_returns = self.__net_returns.sub(risk_free_rate.iloc[1:].values)

        self.__sharpe = np.mean(self.__net_excess_returns) / np.std(self.__net_excess_returns, ddof=1)

    def backtest(self, data, freq_data, volume=pd.DataFrame(), data_type='price', rf=pd.Series(dtype='float'),
                 interval=1, window=60,
                 freq_strategy='D',
                 price_impact=False, ptc_buy=0, ptc_sell=0, ftc=0, c=1, initial_wealth=1E6,
                 extra_data=pd.DataFrame(), price_impact_model='default'):
        """
        Start the backtesting process with the built model. The function itself will not return anything. To get the results,
        please call respective functions.

        :param data: historical data that the strategy to be tested on. Index must be datetime format compatible
        :type data: pd.DataFrame

        :param freq_data: The frequency of the data provided, choose between {'D','W','M'}, where 'D' for day,'W' for week and 'M' for month. 'data' must be taken in the smallest unit of respective frequency, e.g. the frequency 'M' means the data is taken at each month
        :type freq_data: str

        :param volume: trading volume of each asset during each period (array of size T*N), or average trading volume for each asset over all periods (N-d array). If passing in as pd.DataFrame, then its index must match that of the data.
        :type volume: pd.DataFrame or list or np.ndarray or pd.Series

        :param data_type: choose from {'price','return','ex_return'} where 'price' stands for price data of assets at each timestamp, 'return' stands for normal percentage return of each asset in each period, 'ex_return' stands for percentage return net of risk-free rate
        :type data_type: str

        :param rf: data for risk-free rate in each period. Note: if 'rf' is passed in as a dataframe or series, the index of 'rf' must match that of 'data'
        :type rf: pd.Series or pd.DataFrame or int or float

        :param interval: number of periods that users want their portfolios to be rebalanced, the unit is based on 'freq_strategy'. e.g. If 'freq_data' is 'D', while 'freq_strategy' is 'M', and 'interval' is 2, then the portfolio will be rebalanced every 2 months using the user-defined portfolio-construction strategy
        :type interval: int

        :param window: length of rolling windows of 'data' wanted to feed into 'strategy' function. e.g. 'window'=60 means each time during rebalancing, past 60 periods of 'data' will be passed into user-defined strategy function
        :type window: int

        :param freq_strategy: The frequency on which the user want to use 'strategy' to rebalance the portfolio, choose between {'D','W','M'}. If "freq_strategy" is different from "freq_data", the library will resample data on "freq_strategy". Note: 'freq_data' should be smaller than 'freq_strategy' with the sequence 'D' < 'W' < 'M'
        :type freq_strategy: str

        :param price_impact: indicate whether to use price-impact model or not
        :type price_impact: bool

        :param ptc_buy: proportional transaction cost of buying each asset, measured in basis point. Can be a Series or array that provide one cost for each asset, or a single variable that stands for universal transaction cost. Note: Cannot be a list, and must not contain provide labels
        :type ptc_buy: pd.Series or np.ndarray or int or float

        :param ptc_sell: proportional transaction cost of selling each asset, measured in basis point. Can be a Series or array that provide one cost for each asset, or a single variable that stands for universal transaction cost. Note: Cannot be a list, and must not contain provide labels
        :type ptc_sell: pd.Series or np.ndarray or int or float

        :param ftc: dollar value of fixed transaction cost of each transaction, measured in one unit of any currency.
        :type ftc: int or float

        :param c: market depth indicators. Can be a Series or array that provide one market depth for each asset, or a single variable that stands for universal market depth. Note: Do NOT provide labels
        :type c: pd.Series or int or np.ndarray or float

        :param initial_wealth: dollar value of initial wealth of testing when 'price-impact' is true or 'ftc'!=0
        :type initial_wealth: int or float

        :param extra_data: extra_data to be passed into 'strategy' only when 'need_extra_data'==True. Note: 1. the datetime index of extra_data must match that of the provided data. 2. change-of-frequency functionality will be suspended if extra data is needed
        :type extra_data: pd.DataFrame

        :param price_impact_model: choose the price impact model you want to use from {'default'} (testing feature, to be built on)
        :type price_impact_model: str

        :return: None
        """
        if price_impact_model not in {'default'}:
            raise Exception('Unknown type of "price_impact_model"!')

        if type(initial_wealth) != int and type(initial_wealth) != float:
            raise Exception('Wrong type of "initial_wealth" given!')

        if type(c) != float and type(c) != int and not isinstance(c, pd.Series) and not isinstance(c.np.ndarray):
            raise Exception("Wrong type of 'c' given!")

        if type(ftc) != int and type(ftc) != float:
            raise Exception("Wrong type of 'ftc' given!")

        if type(ptc_buy) != int and type(ptc_buy) != float and not isinstance(ptc_buy, pd.Series) and not isinstance(
                ptc_buy,
                np.ndarray):
            raise Exception("Wrong type of 'ptc_buy' provided!")
        else:
            ptc_buy /= 10000

        if type(ptc_sell) != int and type(ptc_sell) != float and not isinstance(ptc_sell, pd.Series) and not isinstance(
                ptc_sell,
                np.ndarray):
            raise Exception("Wrong type of 'ptc_sell' provided!")
        else:
            ptc_sell /= 10000

        if type(price_impact) != bool:
            raise Exception("'price_impact' must be a boolean variable")

        if freq_data not in {'D', 'W', 'M'}:
            raise Exception("'freq_data' must be chosen from {'D','W','M'}")

        if freq_strategy not in {'D', 'W', 'M'}:
            raise Exception("'freq_strategy' must be chosen from {'D','W','M'}")

        if freq_data == 'W' and freq_strategy == 'D':
            raise Exception("'freq_data' should be smaller than 'freq_strategy' with the sequence 'D' < 'W' < 'M'")

        if freq_data == 'M' and freq_strategy in {'D', 'W'}:
            raise Exception("'freq_data' should be smaller than 'freq_strategy' with the sequence 'D' < 'W' < 'M'")

        if type(window) != int:
            raise Exception("'window' must be an 'int' variable")

        if type(interval) != int:
            raise Exception("'interval' must be an 'int' variable")

        if initial_wealth == 1E6:
            if price_impact == True or ftc != 0:
                warnings.warn('Using default initial_wealth value @1E6!')

        if self.__need_extra_data == True:
            if isinstance(extra_data, pd.DataFrame) or isinstance(extra_data, pd.Series):
                if extra_data.empty:
                    raise Exception('Please provide extra_data as dataframe')

                try:
                    extra_data.index = pd.to_datetime(extra_data.index)
                except:
                    print(
                        'Invalid index provided in your "extra_data", please make sure that index is in compatible datetime format')

            else:
                raise Exception(
                    '"extra_data" need to be a Series or DataFrame with datetime index corresponding to test data provided')

            # if user-defined strategy need extra_data to operate, the library will NOT provide change of frequency functionality
            if freq_strategy != freq_data:
                raise Exception(
                    'If "extra_data" needed for your strategy, please make sure "freq_strategy" matches "freq_data"!')
            if not extra_data.index.equals(data.index):
                raise IndexError('Index of extra_data and index of data do not match!')

        if (data_type == 'return' or data_type == 'ex_return') and ('price' in self.__involved_data_type):
            raise Exception('"price" data type is involved in your strategy, please provide data with type "price"')

        if isinstance(rf, pd.Series) or isinstance(rf, pd.DataFrame):
            # if rf.empty and (('ex_return' in self.__involved_data_type) or ('return' in self.__involved_data_type)):
            if rf.empty:
                raise Exception(
                    'Please provide risk-free rate! (Set it to 0 if you do not want to consider it. Note that in this case, net_returns and net_excess_returns will be the same)')
            if not rf.index.equals(data.index):
                raise IndexError('Index of "rf" and index of "data" do not match!')
        elif type(rf) == int or type(rf) == float:
            rf = pd.Series([rf] * data.shape[0], index=data.index)
        else:
            raise Exception('Wrong format of "rf" is given.')

        if ftc != 0:
            if data_type != 'price':
                raise Exception('data_type must be "price" when using fixed transaction cost (ftc!=0)')

        # divide into price_impact model and no_price_impact model
        self.__price_impact = price_impact
        frequency_map = {'D': 'Day', 'W': 'Week', 'M': 'Month'}
        if price_impact == False:
            self.__last_test_frequency = f'{interval} {frequency_map[freq_strategy]}'
            self.__test_no_price_impact(data, freq_data, data_type, rf, interval, window, freq_strategy,
                                        ptc_buy, ptc_sell, ftc, initial_wealth, extra_data)
        else:
            if isinstance(volume, pd.DataFrame):
                if not volume.index.equals(data.index):
                    raise Exception('Index of "volume" and "index" of data do not match!')
            elif isinstance(volume, pd.Series) or isinstance(volume, np.ndarray):
                try:
                    volume = pd.DataFrame(volume.reshape(1, -1), columns=data.columns)
                except:
                    print('Check your volume data!')
                volume = pd.concat([volume] * data.shape[0]).set_index(data.index)
            elif isinstance(volume, list):
                try:
                    volume = pd.DataFrame([volume], columns=data.columns)
                except:
                    print('Check your volume data!')
                volume = pd.concat([volume] * data.shape[0]).set_index(data.index)
            else:
                raise Exception('Please provide volume in correct format!')

            if data_type != 'price':
                raise Exception('Must provide "price" type data for price-impact model')
            elif volume.empty:
                raise Exception(
                    'Must provide correct volume of each asset for price-impact model. For specific requirements '
                    'please refer to the description of the function')
            else:
                self.__last_test_frequency = f'{interval} {frequency_map[freq_strategy]}'
                self.__test_price_impact(data, freq_data, data_type, rf, interval, window, freq_strategy,
                                         ptc_buy, ptc_sell, ftc, volume, c, initial_wealth, extra_data,
                                         price_impact_model)

        return

    def get_net_excess_returns(self):
        '''
        Get the net excess returns (net of risk-free rate) and respective dates of the model tested.
        '''
        return self.__net_excess_returns

    def get_net_returns(self):
        '''
        Get the net returns and respective dates of the model tested
        '''
        return self.__net_returns

    def get_sharpe(self):
        '''
        Get the sharpe ratio of the model tested
        '''
        # self.__sharpe = np.mean(self.__net_excess_returns) / np.std(self.__net_excess_returns, ddof=1)
        return self.__sharpe

    def get_turnover(self):
        '''
        Get the average turnover rate of each period as well as total turnover rate over all periods of the model tested
        '''
        print(f"average turnover is: {self.__average_turnover:.5%}")
        print(f"total turnover is: {self.__total_turnover:.5%}")
        return

    def get_ceq(self, x=1):
        '''
        Get certainty equivalent returns (ceq) of the model tested with the given risk aversion factor
        :param x: risk aversion factor
        :type x: float or int or pd.Series or np.ndarray

        :return: certainty equivalent returns

        '''
        self.__ceq = np.mean(self.__net_excess_returns) - x / 2 * np.cov(self.__net_excess_returns, ddof=1)
        return self.__ceq

    def get_portfolios(self):
        return self.__last_test_portfolios

    def general_performance(self):
        '''
        Get a set of performance evaluation metrics of the model tested
        '''
        output = {}
        output['strategy name'] = self.__name
        output['Price impact'] = 'ON' if self.__price_impact else 'OFF'
        output['Start date of portfolio'] = self.__net_returns.index[0]
        output['End date of portfolio'] = self.__net_returns.index[-1]

        output['Frequency of rebalance'] = self.__last_test_frequency
        output['Duration'] = f'{self.__net_returns.shape[0]} periods'

        evolution = np.cumprod(1 + self.__net_returns)
        output['Final Portfolio Return (%)'] = f"{evolution[-1]:.4%}"
        output['Peak Portfolio Return (%)'] = f"{evolution.max():.4%}"
        output['Bottom Portfolio Return (%)'] = f"{evolution.min():.4%}"

        output['Historical Volatiltiy (%)'] = f"{np.std(self.__net_returns, ddof=1):.4%}"
        output['Sharpe Ratio'] = f"{self.__sharpe:.4f}"

        std_down = np.std(self.__net_excess_returns[self.__net_excess_returns < 0], ddof=1)
        output['Sortino Ratio'] = f"{np.mean(self.__net_excess_returns) / std_down:.4f}"

        drawdown = (evolution.max() - evolution.min()) / evolution.max()
        output['Calmar Ratio'] = f"{np.mean(self.__net_excess_returns) / drawdown:.4f}"
        output['Max. Drawdown (%)'] = f"{drawdown:.4%}"
        output['Max. Drawdown Duration'] = evolution.loc[evolution == evolution.max()].index[0] - evolution.loc[
            evolution == evolution.min()].index[0]

        output[
            '% of positive-net-excess-return periods'] = f"{self.__net_excess_returns[self.__net_excess_returns > 0].count() / self.__net_excess_returns.count():.4%}"
        output[
            '% of positive-net-return periods'] = f"{self.__net_returns[self.__net_returns > 0].count() / self.__net_returns.count():.4%}"

        output['Average turnover (%)'] = f"{self.__average_turnover:.4%}"
        output['Total turnover (%)'] = f"{self.__total_turnover:.4%}"

        output['95% VaR on net-excess returns'] = f"{np.quantile(self.__net_excess_returns, 0.05):.4%}"
        output['95% VaR on net returns'] = f"{np.quantile(self.__net_returns, 0.05):.4%}"

        return pd.Series(output)


# built-in strategies in the library

def __naive_alloc(list_df):
    df = list_df[0]
    n = df.shape[1]
    res = np.ones(n) / n
    return res


naive_alloc = backtest_model(__naive_alloc, ['ex_return'], name='naive allocation portfolio')


def __iv_alloc(list_df):
    # Compute the inverse-variance portfolio
    df = list_df[0]
    cov = df.cov()
    ivp = 1. / np.diag(cov)
    ivp /= ivp.sum()
    return ivp


iv_alloc = backtest_model(__iv_alloc, ['ex_return'], name='inverse variance allocation portfolio')


def __min_var(list_df):
    df = list_df[0]
    n = df.shape[1]
    u = np.ones(n)
    cov = df.cov()
    in_cov = np.linalg.inv(cov)
    w = np.dot(in_cov, u)
    w /= w.sum()
    return w


min_var = backtest_model(__min_var, ['return'], name='min. variance allocation portfolio')


def __mean_variance(list_df):
    df = list_df[0]
    n = df.shape[1]
    cov = df.cov()
    in_cov = np.linalg.inv(cov)
    u = df.mean(axis=0)
    w = np.dot(in_cov, u)
    w /= w.sum()
    return w


basic_mean_variance = backtest_model(__mean_variance, ['ex_return'], name='basic mean-variance allocation portfolio')


def __FF3(list_df, extra_data):
    df = list_df[0]

    X = extra_data
    y = df
    reg = LinearRegression(fit_intercept=True).fit(X, y)
    beta = reg.coef_
    var_epi = (y - reg.predict(X)).var(axis=0)
    cov = np.dot(np.dot(beta, X.cov()), beta.T) + np.diag(var_epi)

    in_cov = np.linalg.inv(cov)
    n = df.shape[1]
    w = np.dot(in_cov, np.ones(n))
    w /= w.sum()
    return w


FF_3_factor_model = backtest_model(__FF3, ['ex_return'], need_extra_data=True,
                                   name='Fama-French 3-factor model portfolio')


def __hrp_alloc(list_df):
    # Compute the hierarchical-risk-parity portfolio
    x = list_df[0]

    def getIVP(cov, **kargs):
        # Compute the inverse-variance portfolio
        ivp = 1. / np.diag(cov)
        ivp /= ivp.sum()
        return ivp

    def getClusterVar(cov, cItems):
        # Compute variance per cluster
        cov_ = cov.loc[cItems, cItems]  # matrix slice
        w_ = getIVP(cov_).reshape(-1, 1)
        cVar = np.dot(np.dot(w_.T, cov_), w_)[0, 0]
        return cVar

    def getQuasiDiag(link):
        # Sort clustered items by distance
        link = link.astype(int)
        sortIx = pd.Series([link[-1, 0], link[-1, 1]])
        numItems = link[-1, 3]  # number of original items
        while sortIx.max() >= numItems:
            sortIx.index = range(0, sortIx.shape[0] * 2, 2)  # make space
            df0 = sortIx[sortIx >= numItems]  # find clusters
            i = df0.index
            j = df0.values - numItems
            sortIx[i] = link[j, 0]  # item 1
            df0 = pd.Series(link[j, 1], index=i + 1)
            sortIx = sortIx.append(df0)  # item 2
            sortIx = sortIx.sort_index()  # re-sort
            sortIx.index = range(sortIx.shape[0])  # re-index
        return sortIx.tolist()

    def getRecBipart(cov, sortIx):
        # Compute HRP alloc
        w = pd.Series(1, index=sortIx)
        cItems = [sortIx]  # initialize all items in one cluster
        while len(cItems) > 0:
            cItems = [i[j:k] for i in cItems for j, k in ((0, int(len(i) / 2)), (int(len(i) / 2), int(len(i)))) if
                      len(i) > 1]  # bisection
            for i in range(0, len(cItems), 2):  # parse in pairs
                cItems0 = cItems[i]  # cluster 1
                cItems1 = cItems[i + 1]  # cluster 2
                cVar0 = getClusterVar(cov, cItems0)
                cVar1 = getClusterVar(cov, cItems1)
                alpha = 1 - cVar0 / (cVar0 + cVar1)
                w[cItems0] *= alpha  # weight 1
                w[cItems1] *= 1 - alpha  # weight 2
        return w

    def correlDist(corr):
        # A distance matrix based on correlation, where 0<=d[i,j]<=1
        # This is a proper distance metric
        dist = ((1 - corr) / 2.) ** .5  # distance matrix
        return dist

    cov, corr = x.cov(), x.corr()
    # clustering
    dist = correlDist(corr)
    link = sch.linkage(dist, 'single')
    sortIx = getQuasiDiag(link)
    sortIx = corr.index[sortIx].tolist()  # recover labels
    df0 = corr.loc[sortIx, sortIx]  # reorder, quasi-diagonalized correlation matrix
    # allocation
    res = getRecBipart(cov, sortIx)
    return res


hrp_alloc = backtest_model(__hrp_alloc, ['ex_return'], name='hierarchical-risk-parity portfolio')


def __Bayes_Stein(list_df):  # ex_return
    df = list_df[0]
    m = 120
    u_ = df.mean(axis=0)
    n = df.shape[1]
    cov_ = np.dot((df - u_).T, df - u_) / (m - n - 2)
    u_min = np.mean(u_)
    inv_cov = np.linalg.inv(cov_)
    sig = (n + 2) / (m * np.dot(np.dot((u_ - u_min).T, inv_cov), u_ - u_min) + n + 2)
    u_bs = (1 - sig) * u_ + sig * u_min
    w = np.dot(inv_cov, u_bs)
    w /= w.sum()
    return w


Bayes_Stein_shrink = backtest_model(__Bayes_Stein, ['ex_return'], name='Bayes_Stein_shrinkage portfolio')

# A small function that fetch the data included in the library package
from importlib import resources


def fetch_data(file_name):
    '''
    Fetch the specific data file from the library.
    Please make sure the correct suffix is on.
    Please inspect these data files before testing to check the arguments and whether they suit the needs.
    :param file_name: name of the data file you want to get from the library, please include suffix
    :type file_name: str

    :return: specific data files
    '''
    if not isinstance(file_name, str):
        raise Exception('Wrong type of "file_name" given. Must be a string. ')

    try:
        with resources.path("portfolio_backtester.data", file_name) as path:
            return pd.read_csv(path, index_col='Date', parse_dates=True)
    except FileNotFoundError:
        raise FileNotFoundError('No such file. Check your file name!')


if __name__ == '__main__':
    # data=fetch_data('SPSectors.csv')
    # naive_alloc.backtest(data.iloc[:,1:],'M',window=120,interval=2, rf=data.iloc[:,0],data_type='ex_return',freq_strategy='M')
    # Bayes_Stein_shrink.backtest(data.iloc[:,1:],'M',window=120,rf=data.iloc[:,0],data_type='ex_return',freq_strategy='M')
    # basic_mean_variance.backtest(data.iloc[:,1:],'M',window=120,rf=data.iloc[:,0],data_type='ex_return',freq_strategy='M')
    # min_var.backtest(data.iloc[:,1:],'M',window=120,rf=data.iloc[:,0],data_type='ex_return',freq_strategy='M')

    # data=fetch_data('sp_500_prices_v2.csv')
    # data = data.iloc[:, :12]
    # volume=fetch_data('sp_500_volumes_v2.csv')
    # volume = volume.loc[:, data.columns]
    #
    # naive_alloc.backtest(data, 'D', window=10, interval=2, rf=pd.Series([0.01] * data.shape[0], index=data.index),
    #                     data_type='price', freq_strategy='W',
    #                     price_impact=False,
    #                     ptc_buy=0.1, ptc_sell=0.2, ftc=1)

    # naive_alloc.backtest(data, 'D', volume, window=3, interval=2, rf=pd.Series([0.01] * data.shape[0], index=data.index),
    #                      data_type='price', freq_strategy='M',
    #                      price_impact=True,
    #                      ptc_buy=0.1, ptc_sell=0.2, ftc=1, c=pd.Series([1] * data.shape[1]))

    #

    # min_var.backtest(data, 'D', volume, window=120, rf=pd.Series([0.01] * data.shape[0], index=data.index),
    #                     data_type='price', freq_strategy='D',
    #                     price_impact=False,
    #                     ptc_buy=0.1, ptc_sell=0.2, ftc=1)

    # naive_alloc.backtest(data, 'D', volume, window=120, rf=pd.Series([0.01] * data.shape[0], index=data.index),
    #                     data_type='price', freq_strategy='D',
    #                     price_impact=False,
    #                     ptc_buy=0.1, ptc_sell=0.2, ftc=1)

    # extra_data=fetch_data('FF3_monthly_192607-202106.csv')
    # start = '1981-01'
    # end = '2002-12'
    # extra_data = extra_data.loc[start:end]
    # extra_data.index = data.index
    # extra_data = extra_data.astype('float64')
    #
    # FF_3_factor_model.backtest(data.iloc[:, 1:], 'M', window=120, rf=data.iloc[:, 0],
    #                            data_type='ex_return', freq_strategy='M',
    #                            price_impact=False, ptc_buy=0.01 , ptc_sell=0.02 , extra_data=extra_data.iloc[:, :-1])

    # hrp_alloc.backtest(data.iloc[:,1:],'M',window=120,rf=data.iloc[:,0],data_type='ex_return',freq_strategy='M')

    #
    #
    data = fetch_data('sp500-0317.csv')
    df = data.iloc[2400:2600, :5]
    naive = backtest_model(__naive_alloc, ['price'], name='naive allocation portfolio')
    naive.backtest(df, freq_data='D', rf=0,interval=2)

    # return_df = df.pct_change()
    # return_df.dropna(axis=0, how='all', inplace=True)
    # iv = backtest_model(lambda x: wrapper(__iv_alloc, x), ['return'])
    # iv.backtest(return_df, freq_data='D', data_type='return', rf=0)
    pass
