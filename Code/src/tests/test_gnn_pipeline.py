import os
import unittest

from src.pipeline import GNNModelPipeline


class GNNPipelineEndToEndTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        """
        Setup the pipeline, in case multiple tests use it
        """
        if os.path.basename(os.getcwd()) == "Code":
            test_file = "./src/tests/Test_Trans.csv"
            cls.pipeline = GNNModelPipeline(dataset_path=test_file)
        else:
            raise AssertionError(
                "Expected tests to run from 'Code' dir"
            )

    def test_pipeline(self):
        """
        Test the entire pipeline flow
        """
        p = self.pipeline

        p.rename_columns()
        p.drop_duplicates()
        p.check_for_null()
        p.extract_currency_features(get_base_amlworld_data=True)
        p.extract_time_features()
        p.create_unique_ids()
        p.extract_additional_time_features()
        p.cyclical_encoding()
        # p.apply_label_encoding()
        p.apply_one_hot_encoding()

        # Train-test split and...
        p.split_train_test_val()
        p.compute_split_specific_node_features()
        p.scale_node_data_frames()
        edge_feats = ["edge_id"] + list(
            set(p.X_cols) - set(
                [
                    "timestamp_int",
                    "hour_of_day",
                    "is_weekend",
                    "edge_id",
                    "sent_amount",
                    "received_amount",
                ]
            )
        )
        p.split_train_test_val_graph(edge_features=edge_feats)
        p.scale_edge_features(
            edge_features_to_scale=[
                "sent_amount_usd",
                # "timestamp_scaled",
                "time_diff_from",
                "time_diff_to",
                "turnaround_time",
            ],
        )
        p.get_data_loaders()

        # Training setup
        p.initialize_training(
            threshold=0.5,
            epochs=1,
            patience=1,
        )
        p.trainer.train()

        # Interpretability
        p.initialize_explainer(epochs=1)
