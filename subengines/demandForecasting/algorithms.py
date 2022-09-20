import datetime
import pandas as pd
from typing import List, Dict
from prophet import Prophet


def create_result_string(results: List) -> str:
    """Concatenates the results with ';' to a string.

    Args:
        results (List): List of the algorithm results

    Returns:
        str: Concatenated results
    """

    return f"{''.join(str(x) + ';' for x in results)}"[:-1]


# ------------------------------------
# Algorithms
# ------------------------------------
"""Algorithm executor function

    Args:
        planning_object (Dict): Dictionary of the timeseries for one planning object
        alogrithm_name (str): Name of the algortihm
        parameters (Dict): Dictionary of the algorithm parameters
        historical_periods (int): Number of historical periods
        forecast_periods (int): Number of forecast periods
        group_id (int): Group ID which identifies the group for which a result is calculated

    Returns:
        Dict: Dictionry of result strings
"""


def average_forecast(df_train: pd.DataFrame, df_expost: pd.DataFrame, df_pred: pd.DataFrame, algorithm_params: Dict,
                     historical_periods: int, forecast_periods: int, group_id: int, request_id: int,
                     algorithm_name: str) -> Dict:
    """ Use a constant prediction calculated by the average. """

    result = {"group_id": group_id}
    try:
        start = datetime.datetime.now()

        # Expost & Forecast
        mean_value = df_train["HISTORY"].mean()
        expost = historical_periods * [mean_value]
        forecast = forecast_periods * [mean_value]

        # Build result dict
        delta = datetime.datetime.now() - start
        messages = [
            f"Forecast and Expost were set to the average of the timeseries after {delta.total_seconds()} seconds.",
            f"Timeseries showed a mean of {mean_value} and standard deviation of {df_train['HISTORY'].std()})."]

        result.update({"EXPOST": create_result_string(expost),
                       "FORECAST": create_result_string(forecast),
                       "messages": messages
                       })


    except Exception as e:
        result["err_message"] = f"Algorithm {algorithm_name} failed to calculate forecast."

    return result


def prophet_forecast(df_train, df_expost, df_pred, algorithm_params, historical_periods, forecast_periods,
                     group_id: int, request_id: int, algorithm_name: str):
    """ Use prophet to make predictions. """

    result = {"group_id": group_id}

    # Read Parameters & set default
    seasonality_mode = algorithm_params.get("seasonality_mode", "additive")
    uncertainty_samples = algorithm_params.get("uncertainty_samples", 0)

    if seasonality_mode not in ["additive", "multiplicative"]:
        result[
            "err_message"] = f"Value '{seasonality_mode}' maintained in forecast model as seasonality_mode must be 'additive' or 'multiplicative'."
        return result

    try:
        uncertainty_samples = int(uncertainty_samples)
    except:
        result[
            "err_message"] = f"Value '{uncertainty_samples}' maintained in forecast model as uncertainty_samples must be integer."
        return result

    # Build model & Calculate forecast
    try:
        # Format df to match expectations of prophet (ds: date/-time, y: timeseries)
        df_train_formatted = df_train.rename(columns={"HISTORY": "y", "timestamp": "ds"})

        # Initiate and train model
        start = datetime.datetime.now()
        m = Prophet(seasonality_mode=seasonality_mode, uncertainty_samples=uncertainty_samples)
        m.fit(df_train_formatted)
        train_time_in_seconds = (datetime.datetime.now() - start).total_seconds()

        # make predictions
        timestamps = df_train["timestamp"].tolist() + df_pred["timestamp"].tolist()
        future = pd.DataFrame({"ds": timestamps})
        predictions = m.predict(future)
        delta_in_seconds = (datetime.datetime.now() - start).total_seconds()

        # Expost & Forecast
        expost = predictions[:historical_periods].yhat
        forecast = predictions[historical_periods:].yhat

        # Result messages
        messages = [
            f"Dates of expost: {timestamps[0].strftime('%Y-%m-%d')}, ..., {timestamps[historical_periods - 1].strftime('%Y-%m-%d')}.",
            f"Dates of forecast: {timestamps[historical_periods].strftime('%Y-%m-%d')}, ..., {timestamps[-1].strftime('%Y-%m-%d')}.",
            f"Prophet used seasonality mode '{m.seasonality_mode}'.",
            f"Runtime for Prophet: {delta_in_seconds:.4f} seconds (Training: {train_time_in_seconds:.4f}, Prediction: {delta_in_seconds - train_time_in_seconds:.4f})."]

        if uncertainty_samples > 0:
            expost_average_uncertainty_range = (
                    predictions[:historical_periods].yhat_upper - predictions[:historical_periods].yhat_lower).mean()
            forecast_average_uncertainty_range = (
                    predictions[historical_periods:].yhat_upper - predictions[historical_periods:].yhat_lower).mean()
            messages[0:0] = [f"Average uncertainty range of Expost: {expost_average_uncertainty_range}.",
                             f"Average uncertainty range of Forecast: {forecast_average_uncertainty_range}."]

        # Build final result
        result.update({"EXPOST": create_result_string(expost),
                       "FORECAST": create_result_string(forecast),
                       "messages": messages
                       })

    except Exception as e:
        result["err_message"] = f"Algorithm {algorithm_name} failed to calculate forecast. Error: {e}"

    return result
