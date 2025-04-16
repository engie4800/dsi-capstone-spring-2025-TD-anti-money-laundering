from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pipeline import ModelPipeline

class Checker:

    def currency_columns_required(pipeline: "ModelPipeline") -> None:
        if (
            "sent_currency" not in pipeline.df.columns or
            "received_currency" not in pipeline.df.columns
        ):
            raise KeyError(
                "Currency columns missing. Need to run 'rename_columns' "
                "preprocessing step first."
            )

    def timestamp_required(pipeline: "ModelPipeline") -> None:
        if "timestamp" not in pipeline.df.columns:
            raise KeyError(
                "Missing 'timestamp' column, were columns renamed properly?"
            )

    def columns_were_renamed(pipeline: "ModelPipeline") -> None:
        if not pipeline.preprocessed["renamed"]:
            raise RuntimeError(
                "This method must be run after renaming columns."
            )

    def time_features_were_extracted(pipeline: "ModelPipeline") -> None:
         if not pipeline.preprocessed["time_features_extracted"]:
            raise RuntimeError(
                "This method must be run after extracting time features."
            )
