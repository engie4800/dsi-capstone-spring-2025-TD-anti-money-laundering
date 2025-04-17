"""To keep the model pipeline class *somewhat* clean in terms of the
volume of code and set of methods on the class, this module contains
feature-identifying functions that accept a dataframe and return the
same dataframe with the addition of the feature.

Note that these methods will need to make assumptions about the data
that is available in the dataframe. Helping to ensure that the data
is available is up to the pipeline, to keep these functions simple (to
let them only worry about adding features).
"""
import numpy as np
import pandas as pd


def add_currency_changed(df: pd.DataFrame) -> pd.DataFrame:
    """Adds the `currency_changed` feature, which is an indicator of
    whether the currency changes within each transaction
    """
    df["currency_changed"] = (
        df["sent_currency"] != df["received_currency"]
    ).astype(int)
    return df


def add_day_of_week(df: pd.DataFrame) -> pd.DataFrame:
    """Adds the `day_of_week` feature, which is an integer representing
    which day of the week (0, 1, 2, ..., 6) the transaction occurs
    """
    df["day_of_week"] = df["timestamp"].dt.weekday
    return df


def add_hour_of_day(df: pd.DataFrame) -> pd.DataFrame:
    """Adds the `hour_of_day` feature, which is an integer representing
    which hour during the day (0, 1, 2, ..., 23) the transaction occurs
    """
    df["hour_of_day"] = df["timestamp"].dt.hour
    return df


def add_is_weekend(df: pd.DataFrame) -> pd.DataFrame:
    """Adds the `is_weekend` feature, which is an indicator of whether
    the transaction occurs during the weekend (1) or week (0)
    """
    df["is_weekend"] = df["day_of_week"].isin([5, 6]).astype(int)
    return df


def add_timestamp_int(df: pd.DataFrame) -> pd.DataFrame:
    """Adds the `timestamp_int` feature, the Unix time at which the
    transaction occurred
    """
    df["timestamp_int"] = df["timestamp"].astype("int64") / 10**9
    return df


def add_timestamp_scaled(df: pd.DataFrame) -> pd.DataFrame:
    """Prefills the `timestamp_scaled` feature as the Unix time, it
    will be scaled later
    """
    # TODO: can we just scale it now? Or, can we avoid adding it and
    # use `timestamp_int` when we scale it later?
    df["timestamp_scaled"] = df["timestamp"].astype("int64") / 10**9
    return df


def add_received_amount_usd(df: pd.DataFrame, usd_conversion: dict) -> pd.DataFrame:
    """Adds the `received_amount_usd` feature, which is the received
    amount converted to USD
    """
    df["received_amount_usd"] = df.apply(
        lambda row: row["received_amount"] * usd_conversion.get(row["received_currency"], 1),
        axis=1,
    )
    return df


def add_seconds_since_midnight(df: pd.DataFrame) -> pd.DataFrame:
    """Adds the `seconds_since_midnight` feature, which is the number
    of seconds that have elapsed since midnight on the day the
    transaction occurred on
    """
    df["seconds_since_midnight"] = (
        df["timestamp"].dt.hour * 3600 +  # Convert hours to seconds
        df["timestamp"].dt.minute * 60 +  # Convert minutes to seconds
        df["timestamp"].dt.second         # Keep seconds
    )
    return df


def add_sent_amount_usd(df: pd.DataFrame, usd_conversion: dict) -> pd.DataFrame:
    """Adds the `sent_amount_usd` feature, which is the sent amount
    converted to USD
    """
    df["sent_amount_usd"] = df.apply(
        lambda row: row["sent_amount"] * usd_conversion.get(row["sent_currency"], 1),
        axis=1,
    )
    return df


def add_time_diff_from(df: pd.DataFrame) -> pd.DataFrame:
    """Adds the `time_diff_from` feature, which is the time elapsed
    since the sender in a given transaction previously sent money
    """
    # Ensures data is sorted by timestamp
    df = df.sort_values(
        by=["from_account_idx", "timestamp_int"]
    ).reset_index(drop=True)

    # Computes the `time_diff_from`, replacing `nan` values with `-1`
    # to represent missing values
    df["time_diff_from"] = df.groupby("from_account_idx")["timestamp_int"].diff()
    df["time_diff_from"] = df["time_diff_from"].fillna(-1)

    # Sort by `edge_id`, which is how the dataframe should have been
    # sorted prior to adding this feature
    df = df.sort_values(by="edge_id").reset_index(drop=True)

    return df


def add_turnaround_time(df: pd.DataFrame) -> pd.DataFrame:
    """Adds the `turnaround_time` feature, which is the time elapsed
    since the sender in a given transaction has received money
    """
    last_received_time = {}
    turnaround_times = []
    for _, row in df.iterrows():
        from_id = row["from_account_idx"]
        to_id = row["to_account_idx"]
        timestamp = row["timestamp_int"]

        # Defines the turnaround time, uses `np.nan` to represent the
        # case where the sender has not previously received money
        turnaround_time = timestamp - last_received_time.get(from_id, np.nan)
        turnaround_times.append(turnaround_time)

        # Adds or updates last received time for the receiver
        last_received_time[to_id] = timestamp

    # Uses `-1` instead of `np.nan` to indicate missing turnaround time
    df["turnaround_time"] = turnaround_times
    df["turnaround_time"] = df["turnaround_time"].fillna(-1)

    return df
