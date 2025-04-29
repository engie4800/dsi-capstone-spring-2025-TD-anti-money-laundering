from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pipeline import BaseModelPipeline

class Checker:

    def currency_columns_required(pipeline: "BaseModelPipeline") -> None:
        if (
            "sent_currency" not in pipeline.df.columns or
            "received_currency" not in pipeline.df.columns
        ):
            raise KeyError(
                "Currency columns missing. Need to run 'rename_columns' "
                "preprocessing step first."
            )

    def timestamp_required(pipeline: "BaseModelPipeline") -> None:
        if "timestamp" not in pipeline.df.columns:
            raise KeyError(
                "Missing 'timestamp' column, were columns renamed properly?"
            )

    def columns_were_renamed(pipeline: "BaseModelPipeline") -> None:
        if not pipeline.preprocessed["renamed"]:
            raise RuntimeError(
                "This method must be run after renaming columns."
            )

    def time_features_were_extracted(pipeline: "BaseModelPipeline") -> None:
         if not pipeline.preprocessed["time_features_extracted"]:
            raise RuntimeError(
                "This method must be run after extracting time features."
            )

    def unique_ids_were_created(pipeline: "BaseModelPipeline") -> None:
        if not pipeline.preprocessed["unique_ids_created"]:
            raise RuntimeError(
                "This method requires each financial entity (each "
                "distinct  bank, account pair) to have a unique "
                "identifier."
            )

    def data_split_to_train_val_test(pipeline: "BaseModelPipeline") -> None:
        if not pipeline.preprocessed["train_test_val_data_split"]:
            raise RuntimeError(
                "Data must have been split into train, test, validation "
                "sets before running this method."
            )

    def train_val_test_node_features_added(pipeline: "BaseModelPipeline") -> None:
        if not pipeline.preprocessed["post_split_node_features"]:
            raise RuntimeError(
                "This method requires node-specific features to have "
                "been added to the pipeline."
            )

    def graph_data_split_to_train_val_test(pipeline: "BaseModelPipeline") -> None:
        if not pipeline.preprocessed["train_test_val_data_split_graph"]:
            raise RuntimeError(
                "Graph data must have been split into train, test, "
                "validation sets before running this method."
            )
