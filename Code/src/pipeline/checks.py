from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pipeline import ModelPipeline

class Checker:

    def currency_columns_required(pipeline: "ModelPipeline") -> bool:
        if (
            "sent_currency" not in pipeline.df.columns or
            "received_currency" not in pipeline.df.columns
        ):
            raise KeyError(
                "Currency columns missing. Need to run 'rename_columns' "
                "preprocessing step first."
            )
