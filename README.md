# portfolio_backtester

- ***portfolio_backtester* is a Python library for backtesting *built-in* or *user-defined* portfolio construction strategies.**

Given a portfolio construction strategy (a function that takes in stock-related data and returns portfolio weights), be 
it pre-built-in or user-defined, and the data that the user wish the strategy to be tested on, the library can calculate several evaluation metrics of the portfolio, including
net returns, sharpe ratio, certainty equivalent returns, turnover, etc.

- ***portfolio_backtester* is fast and efficient, compatible with most `dataframe` and `numpy` objects and functions**

The library is built on `numpy` and `pandas`, and mostly utilizes matrix operations to handle heavy calculations
to make the library *light, fast* and *efficient*.

- ***portfolio_backtester* is flexible and versatile**

Various inputs can be modified to suit the needs of different strategies and backtesting scenarios, 
such as *change of frequency* to enable testing on different *timescales*, *price-impact models* and
*transaction costs* to gauge the impact of trading activities on the strategy, 
ability to take in *extra data* and *trace back prior portfolios* 
to meet the need of portfolio construction strategy, etc.

- ***portfolio_backtester* is setting a universal standard for portfolio performance evaluation**

Since the library is flexible enough to be compatible with most portfolio construction strategies, it aims 
to provide benchmark performance evaluation for all portfolio construction
strategies, so that users can consistently compare the performance of different strategies
under varoius scenarios, without worrying about the possibility of cheating, 
e.g. using *future information*to build portfolio in the current period. By using one universal 
standard, the library enforces a fair game for different strategies by competing on a same level
of battleground.

## Requirements
The following libraries are required for the use of ***portfolio_backtester***
- [numpy](https://numpy.org/)
- [pandas](https://pandas.pydata.org/)
- [scipy](https://www.scipy.org/)
- [scikit-learn](https://scikit-learn.org/stable/)

## Installation

Use the package manager [pip](https://pip.pypa.io/en/stable/) to install portfolio_backtester.

```bash
pip install portfolio_backtester
```

## Get started

### 1. Preparation

Before calling the library, user should decide whether he/she wants to use built-in portfolio construction strategies, or 
defines one himself/herself. If it is the latter, it needs to be realised in Python function with signature
```python
def strategy_name(list_df, extra_data, historical_portfolio): 
    pass
    return w
```
where `list_df` is the list of dataframes that the strategy works with in each period, the sequence of which
should corresponds to `involved_data_type` in the model initialization step later in this section.
`extra_data` and `historical_portfolio` are optional arguments if the strategy needs extra data
or need to refer to past portfolios to construct a new portfolio for the current period. The function should
return a portfolio weight allocation, denoted by `w` here, that **sums up to 1**, in the form of `np.ndarray` or 
`pd.Series` or `pd.DataFrame`.

>Note 1: The function must only return ***ONE*** weight allocation for each period.
> 
>Note 2: The sequence of assets in the return of the function should be consistent with the sequence of assets
> in the data on which the strategy is to be tested throughout the whole process. 

Other than the strategy function, the user also needs to prepare the data for the strategy to
be tested on, as well as the extra data for specific strategies to function, if applicable. 

>Note 1: The library provides several built-in datasets for users' reference. Please use `fetch_data`
> function to gain access to these data
> 
>Note 2: If the strategy requires extra data to function, make sure that the index of extra data matches that 
> of the data for the strategy to be tested on.

### 2. Initialization

After preparation step, the user must first build the model using `backtest_model` from the library. It should
follow the form of:
```python
from portfolio_backtester import backtest_model

strategy_model=backtest_model(strategy_name, involved_data_type, 
                              need_extra_data=False, trace_back=False, name='Unnamed')
```
where `strategy_name` and `involved_data_type` are necessary arguments, `need_extra_data` and `trace_back`
and `name` are optional arguments. `strategy_name` is the portfolio construction function defined in step 1.
`involved_data_type` is a *list* of *str* variables that indicate the *types* of dataframes that the 
strategy function uses, the *str* variables are chosen from `{'price','return','ex_return'}`. 

As mentioned above, the sequence in `involved_data_type` should match that of the `list_df` in the portfolio construction function
`strategy_name`. For example, if `list_df=[price_data, return_data, ex_return_data]` in `strategy_name` from 
preparation, then `involved_data_type=['price','return','ex_return']` in `backtest_model` from initialization.

>Note: The library has some built-in ready-to-be-tested models for users' reference. 

Now it is all set up! The model object is ready to be tested on selected data.

### 3. Test

To test the model on selected data, simply call the function `backtest` on the model object
```python
strategy_model.backtest(data, freq_data, 
                        volume=pd.DataFrame(), data_type='price', rfr=pd.Series(dtype='float'),
                        interval=1, window=60,
                        freq_strategy='D',
                        price_impact=False, tc_a=0, tc_b=0, tc_f=0, c=1, initial_wealth=1E6,
                        extra_data=pd.DataFrame(), price_impact_model='default')
```
although only `data` and `freq_data` are necessary arguments, the user needs to make sure
that 
1. `data_type` actually matches the *type* of data that the model is tested on; 
2. `freq_strategy` matches the frequency at which the user wants the strategy to rebalance the portfolio. 

For available choices and specific requirements of each argument, please refer to the full manual for detailed explaination
and description.

### 4. Results & Analysis

There are several outputs and metrics available. The user can simply call these functions using the model object. A few 
examples are shown below:
```python
strategy_model.general_performance()
strategy_model.get_net_returns()
strategy_model.get_ceq(x=1) # x is the risk aversion factor
```


## Examples
1. **Naive `1/N` strategy**

A simple example with naive `1/N` portfolio construction strategy, no *extra data* needed, do not require
*trace back* of historical portfolios:
```python
# 1. Preparation
# prepare the data
import numpy as np
import pandas as pd
from portfolio_backtester import fetch_data
data = fetch_data('SPSectors.csv')              # We are using built-in datasets in the library

# prepare the portfolio construction function
def __naive_alloc(list_df):
    df = list_df[0]
    n = df.shape[1]
    res = np.ones(n) / n
    return res

# 2. Initialization
from portfolio_backtester import backtest_model

naive_alloc = backtest_model(__naive_alloc, ['ex_return'], name='naive allocation portfolio')

# Note: `naive_alloc` is actually one of the built-in portfolio construction models in the library
# so the user can skip step 1 & 2 by calling in the model from the library directly and go straight to step 3
# The user still needs to prepare the data!
from portfolio_backtester import naive_alloc

# 3. Test
# Most basic version of testing, no change of frequency, price impact not included, no transaction cost, etc.
naive_alloc.backtest(data.iloc[:,1:],'M',window=120,rfr=data.iloc[:,0],
                     data_type='ex_return',freq_strategy='M')
```
Here are a few showcases of results the user can get:
```doctest
strategy name                              naive allocation portfolio
Price impact                                                      OFF
Start date of portfolio                           1991-02-28 00:00:00
End date of portfolio                             2002-12-31 00:00:00
Frequency of rebalance                                        1 Month
Duration                                                  143 periods
Final Portfolio Return (%)                                  422.5701%
Peak Portfolio Return (%)                                   524.5902%
Bottom Portfolio Return (%)                                 107.3055%
Historical Volatiltiy (%)                                     4.1633%
Sharpe Ratio                                                   0.1799
Sortino Ratio                                                  0.2701
Calmar Ratio                                                   0.0094
Max. Drawdown (%)                                            79.5449%
Max. Drawdown Duration                             3745 days 00:00:00
% of positive-net-excess-return periods                      62.9371%
% of positive-net-return periods                             64.3357%
Average turnover (%)                                          3.0853%
Total turnover (%)                                          444.2821%
95% VaR on net-excess returns                                -5.5679%
95% VaR on net returns                                       -5.1337%
dtype: object

>>> naive_alloc.get_net_returns()
Date
1991-02-28    0.073055
1991-03-28    0.030464
1991-04-30   -0.001455
1991-05-31    0.041645
1991-06-28   -0.044282
                ...   
2002-08-30    0.023282
2002-09-30   -0.112382
2002-10-31    0.080945
2002-11-29    0.080691
2002-12-31   -0.031318
Length: 143, dtype: float64

>>> naive_alloc.get_ceq(1)
0.006603752164323863

>>> naive_alloc.get_ceq(np.array([1,2,3]))
array([0.00660375, 0.00574349, 0.00488322])
```

2. **Fama-French 3-factor strategy**

This is a more advanced and sophisticated portfolio construction strategy. It requires *factor data* as *extra
data*, but does not need to *trace back* historical portfolios.

```python
# 1. Preperation
# prepare the data
import numpy as np
import pandas as pd
from portfolio_backtester import fetch_data
data=fetch_data('SPSectors.csv')

extra_data=fetch_data('FF3_monthly_192607-202106.csv')     # extra factor data that is also included in the library
start = '1981-01'
end = '2002-12'
extra_data = extra_data.loc[start:end]
extra_data.index = data.index                      # need to make sure that the index of the two dataframes match

# prepare the portfolio construction function
from sklearn.linear_model import LinearRegression
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

# 2. Initialization
from portfolio_backtester import backtest_model

FF_3_factor_model = backtest_model(__FF3, ['ex_return'], need_extra_data=True,
                                   name='Fama-French 3-factor model portfolio')

# Note: this model is also one of the built-in models in the library, user can call it directly
# Again, user still needs to prepare the data
from portfolio_backtester import FF_3_factor_model

# 3. Test
# Add transaction cost into the backtest
FF_3_factor_model.backtest(data.iloc[:, 1:], 'M', window=120, rfr=data.iloc[:, 0],
                           data_type='ex_return', freq_strategy='M',
                           price_impact=False, ptc_buy=0.01 / 200, ptc_sell=0.01 / 100, 
                           extra_data=extra_data.iloc[:, :-1])
```
And some results are shown below:
```doctest
>>> FF_3_factor_model.general_performance()
strategy name                              Fama-French 3-factor model portfolio
Price impact                                                                OFF
Start date of portfolio                                     1991-02-28 00:00:00
End date of portfolio                                       2002-12-31 00:00:00
Frequency of rebalance                                                  1 Month
Duration                                                            143 periods
Final Portfolio Return (%)                                            259.1316%
Peak Portfolio Return (%)                                             354.4271%
Bottom Portfolio Return (%)                                           105.9731%
Historical Volatiltiy (%)                                               3.5768%
Sharpe Ratio                                                             0.1065
Sortino Ratio                                                            0.1608
Calmar Ratio                                                             0.0054
Max. Drawdown (%)                                                      70.1002%
Max. Drawdown Duration                                       3501 days 00:00:00
% of positive-net-excess-return periods                                58.0420%
% of positive-net-return periods                                       59.4406%
Average turnover (%)                                                   13.3530%
Total turnover (%)                                                   1922.8387%
95% VaR on net-excess returns                                          -5.7882%
95% VaR on net returns                                                 -5.3991%
dtype: object

>>> FF_3_factor_model.get_net_excess_returns()
Date
1991-02-28    0.054931
1991-03-28    0.018295
1991-04-30   -0.020621
1991-05-31    0.010298
1991-06-28   -0.022145
                ...   
2002-08-30    0.025228
2002-09-30   -0.092177
2002-10-31    0.032344
2002-11-29    0.025310
2002-12-31   -0.003465
Length: 143, dtype: float64
```
## Roadmap

In the future:
- More performance metrics 
- More built-in portfolio construction models
- Add in more price-impact model options

## Contributing
Pull requests are welcome. For major changes, please open an issue first to discuss what you would like to change.


## License
[MIT](https://choosealicense.com/licenses/mit/)