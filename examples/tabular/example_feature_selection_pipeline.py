"""Example of a feature selector in tabular data.
"""

from __future__ import annotations

from enum import Enum

import pandas as pd

from autogluon.tabular import TabularDataset, TabularPredictor


class PipelinePosition(str, Enum):
    """Define the positions in the pipeline where custom feature generators can be inserted."""

    START = "start"
    AFTER_NUMERIC_FEATURES = "after_numeric_features"
    AFTER_CATEGORICAL_FEATURES = "after_categorical_features"
    AFTER_DATETIME_FEATURES = "after_datetime_features"
    AFTER_TEXT_SPECIAL_FEATURES = "after_text_special_features"
    AFTER_TEXT_NGRAM_FEATURES = "after_text_ngram_features"
    AFTER_VISION_FEATURES = "after_vision_features"

def run_example():
    train_data = TabularDataset(
        'https://autogluon.s3.amazonaws.com/datasets/AdultIncomeBinaryClassification/train_data.csv')
    test_data = TabularDataset(
        'https://autogluon.s3.amazonaws.com/datasets/AdultIncomeBinaryClassification/test_data.csv')
    predictor = TabularPredictor(
        label="class", path="./ag_path", eval_metric="roc_auc"
    ).fit(
        train_data=train_data,
        hyperparameters={"GBM": {"num_boost_round": 10}},
        num_bag_folds=2,
        num_bag_sets=1,
        verbosity=4,
        dynamic_stacking=False,
        fit_weighted_ensemble=False,
        ag_args_ensemble={"fold_fitting_strategy": "sequential_local"},
        _feature_generator_kwargs={
            "enable_feature_selection": True,
        },
        raise_on_no_models_fitted=False
    )

    predictor.leaderboard(data=test_data, display=True)

    X, y = predictor.load_data_internal()
    with pd.option_context(
        "display.max_rows",
        None,
        "display.max_columns",
        None,
        "display.max_colwidth",
        None,
    ):
        print(X.head())
        print(y.head())


if __name__ == "__main__":
    run_example()
