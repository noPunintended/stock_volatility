import numpy as np
import pandas as pd
import math


def cal_bid_spread(bid_price: float, offer_price: float) -> float:
    ''' Function to calculate bid spread

    Bid spread use to calculate ratio of best offer price and best bid price
    0 is the lowest spread

    Args:
        bid_price: Highest price of bidding
        offer_price: Lowest price of offer

    Returns:
        bid spread
    '''

    bid_spread = (offer_price / bid_price) - 1

    return bid_spread


def cal_wap(bid_price: float, bid_size: float, offer_price: float, offer_size: float) -> float:
    ''' Function to calculate weighted averaged price

    weighted averaged price (wap) is a metric to calculate stock price

    Args:
        bid_price: Highest price of bidding
        bid_size: Size of highest bidding price
        offer_price: Lowest price of offer
        offer_size: Size of lowest offering price

    Returns:
        weighted averaged price
    '''

    wap = ((bid_price * offer_size) + (offer_size * bid_size)) / \
        (bid_size + offer_size)

    return wap


def cal_series_wap(book_df: pd.DataFrame) -> pd.DataFrame:
    ''' Function to calculateb weighted averaged price of DataFrame

    weighted averaged price (wap) is a metric to calculate stock price

    Args:
        book_df: DataFrame contains booking information

    Returns:
        DataFrame contains booking information with wap
    '''

    book_df.loc[:, "wap"] = (book_df['bid_price1'] * book_df['ask_size1']
                             + book_df['ask_price1'] * book_df['bid_size1']) \
        / (book_df['bid_size1'] + book_df['ask_size1'])

    return book_df


def cal_log_return(stock_price_t1: float, stock_price_t2: float) -> float:
    ''' Function to calculate log return

    Log return to calculate return of each stock in two point in time.
    Log rerturn is more mathematically desirable compare to raw return

    Args:
        stock_price_t1: stock price at time 1
        stock_price_t2: stock price at time 2

    Returns:
        log returns
    '''

    log_returns = math.log(stock_price_t2 / stock_price_t1)

    return log_returns


def cal_series_log_return(book_df: pd.DataFrame) -> pd.DataFrame:
    ''' Function to calculate log return

    Log return to calculate return of each stock in two point in time.
    Log rerturn is more mathematically desirable compare to raw return

    Args:
        book_df: DataFrame contains booking information

    Returns:
        DataFrame contains booking information with wap and log return
    '''

    book_df = cal_series_wap(book_df)
    book_df.loc[:, "log_return"] = np.log(book_df['wap']).diff()

    return book_df


def cal_realized_volatility(log_return_series: pd.Series) -> float:
    ''' Function to calculate realized_volatility

    Calculate volatility between time 1 and time 2

    Args:
        log_return_series: a series with log returns

    Returns:
        realized_volatility
    '''

    log_return_series = log_return_series.dropna()
    realized_volatility = np.sqrt(np.sum(np.square(log_return_series)))

    return realized_volatility


def root_mean_squared_percentage_error(y_true: np.array, y_pred: np.array) -> float:
    ''' Function to calculate root mean square percentage error

    Calculate root mean square percentage error between two array

    Args:
        y_true: array of actual value
        y_pred: array of prediction by estimator(s)

    Returns:
        root mean square percentage error
    '''

    rmspe = np.sqrt(np.mean(np.square((y_true - y_pred) / y_true)))

    return rmspe


def symmetric_mean_absolute_percentage_error(y_true: np.array, y_pred: np.array) -> float:
    ''' Function to calculate symmetric mean absolute percentage error

    Calculate symmetric mean absolute percentage error between two array

    Args:
        y_true: array of actual value
        y_pred: array of prediction by estimator(s)

    Returns:
        symmetric mean absolute percentage error
    '''

    smape = 100/len(y_true) * np.sum(
        2 * np.abs(y_pred - y_true) / (np.abs(y_true) + np.abs(y_pred)))

    return smape
