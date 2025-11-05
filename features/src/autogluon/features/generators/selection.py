import logging

import numpy as np
from autogluon.common.features.types import R_INT, R_FLOAT, R_OBJECT, S_TEXT_SPECIAL
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
        self._select_best = None
        self._y = None


    def _fit_transform(self, X: DataFrame, y: Series, **kwargs) -> tuple[DataFrame, dict]:
        self._y = y

        self._select_best_kwargs = {"score_func": chi2, "k": 3}
        self._select_best = SelectKBest(**self._select_best_kwargs).set_output(transform="pandas")
        X_out = self._transform(X, is_train=True)

        selected_features = list(X_out.columns)
        self.feature_metadata_in.keep_features(selected_features, inplace=True)
        type_family_groups_special = {}

        return X_out, type_family_groups_special


    def _transform(self, X: DataFrame, *, is_train: bool = False) -> DataFrame:
        if is_train:
            X = self._select_best.fit_transform(X, self._y)
        else:
            X = self._select_best.transform(X)
        return X


    @staticmethod
    def get_default_infer_features_in_args() -> dict:
        return dict(valid_raw_types=[R_INT, R_FLOAT, R_OBJECT])
