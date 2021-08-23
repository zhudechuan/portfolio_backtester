# portfolio_backtester

- ***portfolio_backtester* is a Python library for backtesting *built-in* or *user-defined* portfolio construction strategies.**

Given a portfolio construction strategy (a function that takes in stock-related data and returns portfolio weights), be 
it pre-built-in or user-defined, and the data that the user wish the strategy to be tested on, the library can calculate several evaluation metrics of the portfolio, including
net returns, sharpe ratio, certainty equivalent returns, turnover, etc.

- ***portfolio_backtester* is fast and efficient, compatible with most `dataframe` and `numpy` objects
and functions**

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

## Usage
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
`extra_data` and `historical_portfolio` are optional arguments if the strategy need extra data
or need to refer to past portfolios to construct a new portfolio for the current period. The function should
return a portfolio weight allocation, denoted by `w` here, that **sums up to 1**, in the form of `np.ndarray` or 
`pd.Series` or `pd.DataFrame`.

>Note 1: The function must only use ***ONE*** period of data and return ***ONE*** weight allocation for that
> period.
> 
>Note 2: The sequence of assets in the return of the function should be consistent with the sequence of assets
> in the data on which the strategy is to be tested throughout the whole process. 

Other than the strategy function, the user also needs to prepare the data for the strategy to
be tested on, as well as the extra data for specific strategies to function, if applicable. 

>Note 1: The library provides several built-in datasets for users' reference.
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
strategy_model.backtest(data, frequency_data, 
                        volume=pd.DataFrame(), data_type='price', rfr=pd.Series(dtype='float'),
                        interval=1, window=60,
                        frequency_strategy='D',
                        price_impact=False, tc_a=0, tc_b=0, tc_f=0, c=1, initial_wealth=1E6,
                        extra_data=pd.DataFrame(), price_impact_model='default')
```
although only `data` and `frequency_data` are necessary arguments, the user needs to make sure
that 
1. `data_type` actually matches the *type* of data that the model is tested on; 
2. `frequency_strategy` matches the frequency at which the user wants the strategy to rebalance the portfolio. 

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

The following shows a simple example with naive `1/N` portfolio construction strategy:
```python
def __naive_alloc(list_df):
    df = list_df[0]
    n = df.shape[1]
    res = np.ones(n) / n
    return res
```


 
Continuing the above example with naive `1/N` portfolio construction strategy:
```python
from portfolio_backtester import backtest_model

# build the model with the strategy function
naive_alloc = backtest_model(__naive_alloc, ['ex_return'], name='naive allocation portfolio')
```

## Roadmap
## Contributing
Pull requests are welcome. For major changes, please open an issue first to discuss what you would like to change.

Please make sure to update tests as appropriate.

## License
[MIT](https://choosealicense.com/licenses/mit/)