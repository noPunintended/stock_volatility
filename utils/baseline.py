import sys
from sklearn import metrics
sys.path.append('../../stock_volatility')
from typing import Dict
import numpy as np
import pandas as pd
from utils.metrics_calculation import (
    cal_series_log_return, 
    cal_realized_volatility
)


def computing_realized_volatility_per_id(book_df: pd.DataFrame) -> pd.Series:
    ''' Function to calculate realized_volatility per stock and times id

    Calculate realized_volatility per stock and times id

    Args:
        book_df: DataFrame contains booking information

    Returns:
        volatility_per_id
    '''

    book_df = cal_series_log_return(book_df)
    volatility_per_id = book_df.groupby(by=["time_id", "stock_id"])["log_return"].agg(
        cal_realized_volatility).rename("volatility_per_id").reset_index()

    return volatility_per_id


def create_baseline_pred(book_df: pd.DataFrame, response_df: pd.DataFrame) -> pd.Series:
    ''' Use volatility_per_id as baseline prediction

    Assume that volatility to be autocorrelated. 
    We can use this property to implement a naive model 
    that just "predicts" realized volatility by using whatever the realized 
    volatility was in the initial 10 minutes.

    Args:
        book_df: DataFrame contains booking information
        response_df: DataFrame contains "training" data

    Returns:
        Series that use volatility per id stock as prediction
    '''

    volatility_per_id = computing_realized_volatility_per_id(book_df)
    results = pd.merge(response_df, volatility_per_id, on=["time_id", "stock_id"], how="left")

    return results["volatility_per_id"]
