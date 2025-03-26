
from copy import deepcopy
from typing import Optional, Union

from matplotlib.axes import Axes
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from pipeline import ModelPipeline
from .colors import SOFT_COLORS


def pre_preprocessed_plot_check(p: ModelPipeline) -> None:
    """
    This check should be run at the beginning of plotting methods that
    assume the data isn't fully preprocessed, e.g. that the plots
    should be made on non-normalized data.
    """
    if not p.preprocessed["renamed"]:
        raise RuntimeError(
            "Columns must be renamed before plotting column imbalance."
        )
    if p.preprocessed["normalized"]:
        raise RuntimeError(
            "Data should not be normalized when plotting column imbalance."
        )


def plot_column_imbalance(
    p: ModelPipeline,
    column: str,
    label: str,
    bins: list[Union[int, float]]=[0, 10, 100, 1000, 10000, np.inf],
    axis_in: Optional[Axes]=None,
) -> None:
    pre_preprocessed_plot_check(p)

    # Operate on a deep copy of the dataframe to avoid polluting the
    # model pipeline dataframe during plotting
    df = deepcopy(p.df)

    # Categorical and numerical data are binned differently.
    # NOTE: not all columns generated during preprocessing are
    # supported here, and updates may be necessary for clean plots to
    # be generated for a given column
    if pd.api.types.is_numeric_dtype(df[column]):
        df["binned"] = pd.cut(
            df[column],
            bins=bins,
            include_lowest=True,
        )
        bin_labels = {
            interval: f"{int(interval.left)} - {int(interval.right) if interval.right != np.inf else '∞'}"
            for interval in df["binned"].cat.categories
        }

        df["binned"] = df["binned"].map(bin_labels)
        all_types = sorted(
            bin_labels.values(),
            key=lambda x: int(x.split(" - ")[0]),
        )
        data_column = "binned"
    else:
        if column == "hour_of_day":
            all_types = sorted(df[column].unique(), key=int)
        else:
            all_types = sorted(df[column].unique())
        data_column = column

    df_licit = df[df["is_laundering"] == 0]
    proportion_licit = df_licit[data_column].value_counts(normalize=True) * 100
    proportion_licit = proportion_licit.reindex(all_types, fill_value=0)

    df_illicit = df[df["is_laundering"] == 1]
    proportion_illicit = df_illicit[data_column].value_counts(normalize=True) * 100
    proportion_illicit = proportion_illicit.reindex(all_types, fill_value=0)

    total_proportion = proportion_licit + proportion_illicit
    licit_normalized = (proportion_licit / total_proportion) * 100
    illicit_normalized = (proportion_illicit / total_proportion) * 100

    if axis_in is None:
        _, ax = plt.subplots(figsize=(7, 4))
    else:
        ax = axis_in

    y_pos = np.arange(len(all_types))

    ax.barh(
        y_pos,
        licit_normalized,
        color=SOFT_COLORS["green"],
        label="Licit",
    )
    ax.barh(
        y_pos,
        illicit_normalized,
        left=licit_normalized,
        color=SOFT_COLORS["blue"],
        label="Illicit",
    )
    ax.axvline(50, linestyle="--", color="black", linewidth=1)

    ax.set_xlim((0, 100))
    ax.set_yticks(y_pos)
    ax.set_yticklabels(all_types)
    ax.set_xlabel("Proportion")
    ax.set_title(f"{label}, Licit vs. Illicit")
    if ax.get_legend() is None:
        # Ensure only the first subplot gets a legend if using aggregate
        # plotting method
        ax.legend(loc="upper right", bbox_to_anchor=(1, 1))
    plt.gca().invert_yaxis()

    if axis_in is None:
        plt.show()


def plot_column_imbalances(p: ModelPipeline) -> None:
    """
    Wrapper plot to create a 2x2 plot of base data imbalance
    """
    pre_preprocessed_plot_check(p)

    _, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Converts the 2x2 array to a list for use in iteration
    axes = axes.flatten()

    columns = [
        ("Sent Currency", "sent_currency"),
        ("Received Currency", "received_currency"),
        ("Sent Amount", "sent_amount"),
        ("Received Amount", "received_amount"),
    ]

    for i, (label, column) in enumerate(columns):
        plot_column_imbalance(p, column, label, axis_in=axes[i])

    plt.tight_layout()
    plt.show()
