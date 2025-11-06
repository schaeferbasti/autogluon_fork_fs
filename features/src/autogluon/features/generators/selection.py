import logging

from autogluon.common.features.types import R_INT, R_FLOAT, R_OBJECT
from pandas import DataFrame, Series

from .abstract import AbstractFeatureGenerator

from sklearn.feature_selection import chi2
from sklearn.feature_selection import SelectKBest
from tabarena.benchmark.feature_selection_methods.ag.select_k_best_chi2.select_k_best_chi2 import Select_k_Best_Chi2
from tabarena.benchmark.feature_selection_methods.ag.boruta.boruta import Boruta

logger = logging.getLogger(__name__)

FEATURE_SELECTION_METHODS = {
    "Select_k_Best_Chi2": Select_k_Best_Chi2,
    "Boruta": Boruta
}


class FeatureSelectionGenerator(AbstractFeatureGenerator):
    """FeatureSelectionGenerator selects features from the data."""

    def __init__(self, enable_feature_selection=None, **kwargs):
        super().__init__(**kwargs)
        self._select_best = None
        self._y = None
        self._delegate = None


        # Determine which method to use
        if enable_feature_selection is None:
            self._delegate = None
        elif isinstance(enable_feature_selection, str):
            delegate_class = FEATURE_SELECTION_METHODS.get(enable_feature_selection)
            if delegate_class is None:
                self._delegate = None
            else:
                self._delegate = delegate_class(**kwargs)
        else:
            self._delegate = None


    def _fit_transform(self, X: DataFrame, y: Series, **kwargs) -> tuple[DataFrame, dict]:
        self._y = y

        if self._delegate is not None:
            self._delegate.feature_metadata_in = self.feature_metadata_in
            X_out, type_family_groups_special = self._delegate._fit_transform(X, y, **kwargs)
            self.feature_metadata_in = self._delegate.feature_metadata_in
            return X_out, type_family_groups_special

        self._select_best_kwargs = {"score_func": chi2, "k": 3}
        self._select_best = SelectKBest(**self._select_best_kwargs).set_output(transform="pandas")
        X_out = self._transform(X, is_train=True)

        selected_features = list(X_out.columns)
        self.feature_metadata_in.keep_features(selected_features, inplace=True)
        type_family_groups_special = {}

        return X_out, type_family_groups_special


    def _transform(self, X: DataFrame, *, is_train: bool = False) -> DataFrame:
        if self._delegate is not None:
            return self._delegate._transform(X, is_train=is_train)

        if is_train:
            X = self._select_best.fit_transform(X, self._y)
        else:
            X = self._select_best.transform(X)
        return X


    @staticmethod
    def get_default_infer_features_in_args() -> dict:
        return dict(valid_raw_types=[R_INT, R_FLOAT, R_OBJECT])
