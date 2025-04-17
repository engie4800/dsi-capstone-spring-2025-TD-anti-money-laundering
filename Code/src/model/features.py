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


def add_sent_amount_usd(df: pd.DataFrame, usd_conversion: dict) -> pd.DataFrame:
    """Adds the `sent_amount_usd` feature, which is the sent amount
    converted to USD
    """
    df["sent_amount_usd"] = df.apply(
        lambda row: row["sent_amount"] * usd_conversion.get(row["sent_currency"], 1),
        axis=1,
    )
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
