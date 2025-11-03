import logging

import numpy as np
from autogluon.common.features.feature_metadata import FeatureMetadata
from pandas import DataFrame, Series
from sklearn.compose import ColumnTransformer
from sklearn.feature_selection import chi2
from sklearn.feature_selection import SelectKBest
from sklearn.preprocessing import OneHotEncoder

from .abstract import AbstractFeatureGenerator

logger = logging.getLogger(__name__)


class FeatureSelector(AbstractFeatureGenerator):
    """FeatureSelectionGenerator selects features from the data."""

    def _fit_transform(self, X: DataFrame, y: Series, **kwargs) -> tuple[DataFrame, dict]:
        self._y = y

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

        self._select_best_kwargs = {"score_func": chi2, "k": 1}
        self._select_best = SelectKBest(**self._select_best_kwargs).set_output(transform="pandas")

        X_out = self._transform(X, is_train=True)
        type_group_map_special = self.feature_metadata_in.type_group_map_special

        return X_out, type_group_map_special

    def _transform(self, X: DataFrame, *, is_train: bool = False) -> DataFrame:
        if is_train:
            X = self._select_best.fit_transform(X, self._y)
        else:
            X = self._select_best.transform(X)
        X.columns = [f"__feature_selection_kbest_chi2_{i}" for i in range(X.shape[1])]
        return X

    @staticmethod
    def get_default_infer_features_in_args() -> dict:
        return dict()

    def _more_tags(self):
        return {"feature_interactions": False}

    def estimate_output_feature_metadata(self, feature_metadata_in: FeatureMetadata) -> FeatureMetadata:
        features_to_remove = feature_metadata_in.get_features(**self._infer_features_in_args)
        return feature_metadata_in.keep_features(features_to_remove, inplace=False)

