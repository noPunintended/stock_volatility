import sys

from sklearn import metrics
sys.path.append('../../stock_volatility')
from typing import Dict
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from utils.metrics_calculation import (
    cal_series_log_return, 
    cal_realized_volatility, 
    root_mean_squared_percentage_error,
    symmetric_mean_absolute_percentage_error
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
        lambda x: cal_realized_volatility(x)).rename("volatility_per_id").reset_index()

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
    results = pd.merge(response_df, volatility_per_id, on=["stock_id", "time_id"], how="left")

    return results["volatility_per_id"]


def evaluation(y_true: np.array, y_pred: np.array) -> dict:
    ''' Function to calculate metrics for models evaluation
    Calculate metrics for evaluate regression including:
    residule, sum_error, mae, rmse, rmspe, smape, r2

    Args:
        y_true: array of actual value
        y_pred: array of prediction by estimator(s)

    Returns:
        Dictionary of metrics
    '''

    residule = y_true - y_pred
    # This call tell if the estimator over/under-predicted
    sum_error = np.sum(residule)  
    mae = mean_absolute_error(y_true, y_pred)
    rmse = mean_squared_error(y_true, y_pred, squared=False)
    rmspe = root_mean_squared_percentage_error(y_true, y_pred)
    smape = symmetric_mean_absolute_percentage_error(y_true, y_pred)

    # Again R2 is really problematic and should not be use in this instance,
    # but the example notebook has them
    r2 = r2_score(y_true, y_pred)

    metrics_result = {
        "residule": residule,
        "sum_error": sum_error,
        "mae": mae,
        "rmse": rmse,
        "rmspe": rmspe,
        "smape": smape,
        "r2": r2
    }

    return metrics_result
