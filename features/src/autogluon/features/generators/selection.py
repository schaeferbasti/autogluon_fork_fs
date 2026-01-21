import logging
import time
from autogluon.common.features.types import R_INT, R_FLOAT, R_OBJECT
from pandas import DataFrame, Series

from .abstract import AbstractFeatureSelector


from tabarena.benchmark.feature_selection_methods.ag.randomfs.RandomFS import RandomFS
from tabarena.benchmark.feature_selection_methods.ag.original.Original import Original
from tabarena.benchmark.feature_selection_methods.ag.enumeration.enumeration_fs import EnumerationFeatureSelector

from tabarena.benchmark.feature_selection_methods.ag.t_test.tTest import tTest
from tabarena.benchmark.feature_selection_methods.ag.rf_importance.RFImportance import RFImportance
from tabarena.benchmark.feature_selection_methods.ag.information_gain.InformationGain import InformationGain
from tabarena.benchmark.feature_selection_methods.ag.jmi.JMI import JMI
from tabarena.benchmark.feature_selection_methods.ag.cmim.CMIM import CMIM
from tabarena.benchmark.feature_selection_methods.ag.mrmr.mRMR import mRMR
from tabarena.benchmark.feature_selection_methods.ag.gini.Gini import Gini
from tabarena.benchmark.feature_selection_methods.ag.relieff.ReliefF import ReliefF
from tabarena.benchmark.feature_selection_methods.ag.lasso.Lasso import Lasso
from tabarena.benchmark.feature_selection_methods.ag.chi2.Chi2 import Chi2
from tabarena.benchmark.feature_selection_methods.ag.laplacian_score.LaplacianScore import LaplacianScore
from tabarena.benchmark.feature_selection_methods.ag.fisher_score.FisherScore import FisherScore
from tabarena.benchmark.feature_selection_methods.ag.disr.DISR import DISR

from tabarena.benchmark.feature_selection_methods.ag.select_k_best_f.select_k_best_f import Select_k_Best_F
from tabarena.benchmark.feature_selection_methods.ag.boruta.boruta import Boruta
from tabarena.benchmark.feature_selection_methods.ag.mafese.MAFESE import MAFESE

from tabarena.benchmark.feature_selection_methods.ag.ls_flip.ls_flip import LocalSearchFeatureSelector_Flip
from tabarena.benchmark.feature_selection_methods.ag.ls_flipswap.ls_flipswap import LocalSearchFeatureSelector_FlipSwap

from tabarena.benchmark.feature_selection_methods.ag.metafs.MetaFS import MetaFS


logger = logging.getLogger(__name__)


FEATURE_SELECTION_METHODS = {
    "Original": Original,
    "RandomFS": RandomFS,
    "Enumeration": EnumerationFeatureSelector,

    # Chosen Filter Methods
    "tTest": tTest,
    "RFImportance": RFImportance,
    "InformationGain": InformationGain,
    "JMI": JMI,
    "CMIM": CMIM,
    "mRMR": mRMR,
    "Gini": Gini,
    "ReliefF": ReliefF,
    "Lasso": Lasso,
    "Chi2": Chi2,
    "LaplacianScore": LaplacianScore,
    "FisherScore": FisherScore,
    "DISR": DISR,

    # Other methods
    "SelectKBest": Select_k_Best_F,
    "Boruta": Boruta,
    "Mafese": MAFESE,

    "LS_Flip": LocalSearchFeatureSelector_Flip,
    "LS_FlipSwap": LocalSearchFeatureSelector_FlipSwap,

    "MetaFS": MetaFS,
}


class FeatureSelector(AbstractFeatureSelector):
    """FeatureSelector selects features from the data."""

    def __init__(self, enable_feature_selection=None, **kwargs):
        super().__init__(**kwargs)
        self._select_best = None
        self._delegate = None
        self._y = None
        self._model = None
        self._n_max_features = None
        self._selected_features = None
        self.method_name = enable_feature_selection

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


    def _fit_transform(self, X: DataFrame, y: Series, model, n_max_features: int, **kwargs) -> tuple[DataFrame, dict]:
        self._y = y
        self._model = model
        self._n_max_features = n_max_features

        if self._delegate is not None:
            self._delegate.feature_metadata_in = self.feature_metadata_in
            X_out, type_family_groups_special = self._delegate._fit_transform(X, y, model, n_max_features, **kwargs)
            self.feature_metadata_in = self._delegate.feature_metadata_in
            return X_out, type_family_groups_special

        logger.warning(f'\tWarning: FeatureSelection Method {self.method_name} not found... Using random feature selection')
        self._random_fs = RandomFS()
        # Time limit
        if "time_limit" in kwargs and kwargs["time_limit"] is not None:
            time_start_fit = time.time()
            kwargs["time_limit"] -= time_start_fit - kwargs["start_time"]
            if kwargs["time_limit"] <= 0:
                logger.warning(
                    f'\tWarning: FeatureSelection Method has no time left to train... (Time Left = {kwargs["time_limit"]:.1f}s)')
                if n_max_features is not None and len(X.columns) > n_max_features:
                    X_out = X.sample(n=n_max_features, axis=1)
                    return X_out
                else:
                    return X
        X_out = self._random_fs.fit_transform(X, y, model, n_max_features, **kwargs)
        if n_max_features is not None and len(X_out.columns) > n_max_features:
            X_out = X_out.sample(n=n_max_features, axis=1)
        self._selected_features = list(X_out.columns)
        type_family_groups_special = {}
        return X_out, type_family_groups_special


    def _transform(self, X: DataFrame, *, is_train: bool = False) -> DataFrame:
        if self._delegate is not None:
            return self._delegate._transform(X, is_train=is_train)

        if is_train:
            X = self._random_fs.fit_transform(X, self._y, self._model, self._n_max_features)
            self._selected_features = list(X.columns)
        else:
            X = X[self._random_fs._selected_features]
        return X


    @staticmethod
    def get_default_infer_features_in_args() -> dict:
        return dict(valid_raw_types=[R_INT, R_FLOAT, R_OBJECT])
