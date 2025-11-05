import logging

import numpy as np
from autogluon.common.features.types import R_INT, R_FLOAT, R_OBJECT
from pandas import DataFrame, Series

from .abstract import AbstractFeatureGenerator

from sklearn.compose import ColumnTransformer
from sklearn.feature_selection import chi2
from sklearn.feature_selection import SelectKBest
from sklearn.preprocessing import OneHotEncoder

logger = logging.getLogger(__name__)


class FeatureSelector(AbstractFeatureGenerator):
    """FeatureSelectionGenerator selects features from the data."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)


    def _fit_transform(self, X: DataFrame, y: Series, **kwargs) -> tuple[DataFrame, dict]:
        self._y = y

        self._select_best_kwargs = {"score_func": chi2, "k": 1}
        self._select_best = SelectKBest(**self._select_best_kwargs).set_output(transform="pandas")
        X_out = self._transform(X, is_train=True)

        features_to_remove = self.feature_metadata_in.get_features(**self._infer_features_in_args)
        self.feature_metadata_in = self.feature_metadata_in.keep_features(features_to_remove, inplace=False)

        return X_out


    def _transform(self, X: DataFrame, *, is_train: bool = False) -> DataFrame:
        if is_train:
            categorical_columns = X.select_dtypes(include=['object', 'category']).columns.tolist()
            numerical_columns = X.select_dtypes(include=[np.number]).columns.tolist()
            if categorical_columns:
                preprocessor = ColumnTransformer(
                    transformers=[
                        ('num', 'passthrough', numerical_columns),
                        ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), categorical_columns)
                    ],
                    remainder='passthrough'
                )
                X_transformed = preprocessor.fit_transform(X)
                num_cols = numerical_columns
                cat_cols = preprocessor.named_transformers_['cat'].get_feature_names_out(categorical_columns).tolist()
                all_cols = num_cols + cat_cols
                X = DataFrame(X_transformed, columns=all_cols, index=X.index)
            X = self._select_best.fit_transform(X, self._y)
        else:
            X = self._select_best.transform(X)
        return X


    @staticmethod
    def get_default_infer_features_in_args() -> dict:
        return dict(valid_raw_types=[R_INT, R_FLOAT, R_OBJECT])
