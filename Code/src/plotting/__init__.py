"""
The plotting module consists of a collection of methods that operate on
Pandas data frames. These are decoupled from the model pipeline to keep
the pipeline module concise. They are meant to operate on the dataframe
attached to the model pipeline, either after the data columns are
renamed, or after the data is preprocessed. Example use:

    p = ModelPipeline(dataset_path="path/to/data.csv")
    p.rename_columns()
    plot_column_imbalance(p, "payment_type", "Payment Type")

"""
from .colors import (
    COLORS,
    SOFT_COLORS,
)
from .column_imbalance import (
    plot_column_imbalance,
    plot_column_imbalances,
)
from .explanation_subraph import plot_explanation_subgraph


__all__ = [
    "COLORS",
    "SOFT_COLORS",
    "plot_column_imbalance",
    "plot_column_imbalances",
    "plot_explanation_subgraph",
]
