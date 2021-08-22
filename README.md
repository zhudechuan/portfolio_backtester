# portfolio_backtester

***portfolio_backtester*** is a Python library for backtesting *built-in* or *user-defined* portfolio construction strategy.

Given a user-defined portfolio construction strategy (a function that takes in stock-related data and returns portfolio weights) and
the data that the user wish the strategy to be tested on, the library can calculate several evaluation metrics of the portfolio, including
net_returns, sharpe ratio, certainty equivalent returns, turnover, etc.

Various inputs can be modified to suit the needs of different strategies and backtesting scenarios, such as *price-impact models*,
*transaction costs*, taking in *extra data* to meet the need of portfolio construction strategy, etc.

## Installation

Use the package manager [pip](https://pip.pypa.io/en/stable/) to install portfolio_backtester.

```bash
pip install portfolio_backtester
```

## Usage Example
### Preparation

Before calling in the library, user should decide whether he/she wants to use built-in portfolio construction strategies, or 
defines one himself/herself. It needs to be realised in Python function with signature
```python
def strategy_name(list_df, extra_data, historical_portfolio)
```
where `extra_data` and `historical_portfolio` are optional arguments if the strategy need extra data
or need to refer to past portfolios to construct a new portfolio for the current period.


### Initialization

```python
import portfolio_backtester

# returns 'words'
foobar.pluralize('word')

# returns 'geese'
foobar.pluralize('goose')

# returns 'phenomenon'
foobar.singularize('phenomena')
```

## Contributing
Pull requests are welcome. For major changes, please open an issue first to discuss what you would like to change.

Please make sure to update tests as appropriate.

## License
[MIT](https://choosealicense.com/licenses/mit/)