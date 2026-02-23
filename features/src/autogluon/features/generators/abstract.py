import copy
import gc
import inspect
import logging
import math
import os
import pickle
import sys
import time
from collections import defaultdict
from typing import Dict, List, Any

import numpy as np
import pandas as pd
from autogluon.common.loaders import load_pkl, load_json
from autogluon.common.space import Space
from autogluon.common.utils.distribute_utils import DistributedContext
from autogluon.common.utils.lite import disable_if_lite_mode
from autogluon.common.utils.resource_utils import ResourceManager, get_resource_manager
from autogluon.common.utils.utils import setup_outputdir
from autogluon.core.constants import AG_ARGS_FIT, AG_ARG_PREFIX
from autogluon.core.data import LabelCleaner
from autogluon.core.utils import infer_problem_type
from autogluon.core.utils.exceptions import NotEnoughMemoryError, TimeLimitExceeded, NumberOfFeaturesError
from pandas import DataFrame, Series

from autogluon.common.features.feature_metadata import FeatureMetadata
from autogluon.common.features.infer_types import get_type_group_map_special, get_type_map_raw, get_type_map_real
from autogluon.common.savers import save_pkl, save_json

from autogluon.common.utils.pandas_utils import get_approximate_df_mem_usage

from ..utils import is_useless_feature

logger = logging.getLogger(__name__)


# TODO: Add option to minimize memory usage of feature names by making them integers / strings of integers
# TODO: Add ability to track which input features created which output features.
# TODO: Add log of # of observation counts to high cardinality categorical features
class AbstractFeatureGenerator:
    """
    Abstract feature generator implementation from which all AutoGluon feature generators inherit.
    The purpose of a feature generator is to transform data from one form to another in a stateful manner.
    First, the generator is initialized with various arguments that dictate the way features are generated.
    Then, the generator is fit through either the `.fit()` or `.fit_transform()` methods using training data typically in pandas DataFrame format.
    Finally, the generator can transform new data with the same initial format as the training data through the `.transform()` method.

    Parameters
    ----------
    features_in : list, default None
        List of feature names the generator will expect and use in the fit and transform methods.
        Any feature in an incoming DataFrame that is not present in features_in is dropped and will not influence the transformation logic.
        If None, infer during fit from the _infer_features_in method.
        Equivalent to feature_metadata_in.get_features() post-fit.
    feature_metadata_in : :class:`autogluon.common.features.feature_metadata.FeatureMetadata`, default None
        :class:`FeatureMetadata` object corresponding to the training data input features.
        If None, infer during fit from the _infer_feature_metadata_in method.
        Any features not present in features_in (if provided) will be removed from feature_metadata_in.
    post_generators : list of FeatureGenerators, default None
        FeatureGenerators which will fit and transform sequentially after this object's transformation logic,
        feeding their output into the next generator's input.
        The output of the final FeatureGenerator will be the used as the transformed output.
    pre_enforce_types : bool, default False
        If True, the exact raw types (int64, float32, etc.) of the training data will be enforced on future data,
        either converting the types to the training types or raising an exception if unable.
        This is important to set to True on the outer feature generator in a feature generation pipeline to ensure
        incorrect dtypes are not passed downstream, but is often redundant when used on inner feature generators inside a pipeline.
    pre_drop_useless : bool, default False
        If True, features_in will be pruned at fit time of features containing only a single unique value across all rows.
    post_drop_duplicates : bool, default False
        If True, a :class:`DropDuplicatesFeatureGenerator` will be appended to post_generators.
        This feature generator will drop any duplicate features found in the data, keeping only one feature within any duplicate feature sets.
        Warning: For large datasets with many features, this may be very computationally expensive or even computationally infeasible.
    reset_index : bool, default False
        If True, for the duration of fit and transform, the input data's index is reset to be monotonically increasing from 0 to N-1 for a dataset of N rows.
        At the end of fit and transform, the original index is re-applied to the output data.
        This is important to set to True on the outer feature generator in a feature generation pipeline to ensure that a non-default
        index does not cause corruption of the inner feature generation if any inner feature generator does not properly handle non-default indices.
        This index reset is also applied to the y label data if provided during fit.
    column_names_as_str : bool, default True
        If True, the column names of the input data are converted to string if they were not already.
        This solves any issues related to downstream FeatureGenerators and models which cannot handle integer column names, and allows
        column name prefix and suffix operations to avoid errors.
        Note that for performance purposes, column names are only converted at transform time if they were not strings at fit time.
        Ensure consistent column names as input to avoid errors.
    name_prefix : str, default None
        Name prefix to add to all output feature names.
    name_suffix : str, default None
        Name suffix to add to all output feature names.
    infer_features_in_args : dict, default None
        Used as the kwargs input to FeatureMetadata.get_features(**kwargs) when inferring self.features_in.
        This is merged with the output dictionary of self.get_default_infer_features_in_args() depending on the value of infer_features_in_args_strategy.
        Only used when features_in is None.
        If None, then self.get_default_infer_features_in_args() is used directly.
        Refer to FeatureMetadata.get_features documentation for a full description of valid keys.
        Note: This is advanced functionality that is not necessary for most situations.
    infer_features_in_args_strategy : str, default 'overwrite'
        Determines how infer_features_in_args and self.get_default_infer_features_in_args() are combined to result in self._infer_features_in_args
        which dictates the features_in inference logic.
        If 'overwrite': infer_features_in_args is used exclusively and self.get_default_infer_features_in_args() is ignored.
        If 'update': self.get_default_infer_features_in_args() is dictionary updated by infer_features_in_args.
        If infer_features_in_args is None, this is ignored.
    banned_feature_special_types : List[str], default None
        List of feature special types to additionally exclude from input. Will update self.get_default_infer_features_in_args().
    log_prefix : str, default ''
        Prefix string added to all logging statements made by the generator.
    verbosity : int, default 2
        Controls the verbosity of logging.
        0 will silence logs, 1 will only log warnings, 2 will log info level information, and 3 will log info level information and provide detailed
        feature type input and output information.
        Logging is still controlled by the global logger configuration, and therefore a verbosity of 3 does not guarantee that logs will be output.

    Attributes
    ----------
    features_in : list of str
        List of feature names the generator will expect and use in the fit and transform methods.
        Equivalent to feature_metadata_in.get_features() post-fit.
    features_out : list of str
        List of feature names present in the output of fit_transform and transform methods.
        Equivalent to feature_metadata.get_features() post-fit.
    feature_metadata_in : FeatureMetadata
        The FeatureMetadata of data pre-transformation (data used as input to fit and transform methods).
    feature_metadata : FeatureMetadata
        The FeatureMetadata of data post-transformation (data outputted by fit_transform and transform methods).
    feature_metadata_real : FeatureMetadata
        The FeatureMetadata of data post-transformation consisting of the exact dtypes as opposed to the grouped raw dtypes found in feature_metadata_in,
        with grouped raw dtypes substituting for the special dtypes.
        This is only used in the print_feature_metadata_info method and is intended for introspection. It can be safely set to None to reduce memory and
        disk usage post-fit.
    """

    def __init__(
            self,
            features_in: list = None,
            feature_metadata_in: FeatureMetadata = None,
            post_generators: list = None,
            pre_enforce_types=False,
            pre_drop_useless=False,
            post_drop_duplicates=False,
            reset_index=False,
            column_names_as_str=True,
            name_prefix: str = None,
            name_suffix: str = None,
            infer_features_in_args: dict = None,
            infer_features_in_args_strategy="overwrite",
            banned_feature_special_types: List[str] = None,
            log_prefix="",
            verbosity=2,
    ):
        self._is_fit = False  # Whether the feature generator has been fit
        self.features_in = features_in  # Original features to use as input to feature generation
        self.features_out = None  # Final list of features after transformation
        self.feature_metadata_in: FeatureMetadata = (
            feature_metadata_in  # FeatureMetadata object based on the original input features.
        )

        # FeatureMetadata object based on the processed features. Pass to models to enable advanced functionality.
        self.feature_metadata: FeatureMetadata = None

        # TODO: Consider merging feature_metadata and feature_metadata_real, have FeatureMetadata contain exact dtypes, grouped raw dtypes,
        #  and special dtypes all at once.
        # FeatureMetadata object based on the processed features, containing the true raw dtype information (such as int32, float64, etc.).
        # Pass to models to enable advanced functionality.
        self.feature_metadata_real: FeatureMetadata = None
        self._feature_metadata_before_post = None  # FeatureMetadata directly prior to applying self._post_generators.
        self._infer_features_in_args = self.get_default_infer_features_in_args()
        if infer_features_in_args is not None:
            if infer_features_in_args_strategy == "overwrite":
                self._infer_features_in_args = copy.deepcopy(infer_features_in_args)
            elif infer_features_in_args_strategy == "update":
                self._infer_features_in_args.update(infer_features_in_args)
            else:
                raise ValueError(
                    f"infer_features_in_args_strategy must be one of: {['overwrite', 'update']}, but was: '{infer_features_in_args_strategy}'"
                )
        if banned_feature_special_types:
            if "invalid_special_types" not in self._infer_features_in_args:
                self._infer_features_in_args["invalid_special_types"] = banned_feature_special_types
            else:
                for f in banned_feature_special_types:
                    if f not in self._infer_features_in_args["invalid_special_types"]:
                        self._infer_features_in_args["invalid_special_types"].append(f)

        if post_generators is None:
            post_generators = []
        elif not isinstance(post_generators, list):
            post_generators = [post_generators]
        self._post_generators: list = post_generators
        if post_drop_duplicates:
            from .drop_duplicates import DropDuplicatesFeatureGenerator

            self._post_generators.append(DropDuplicatesFeatureGenerator(post_drop_duplicates=False))
        if name_prefix or name_suffix:
            from .rename import RenameFeatureGenerator

            # inplace=False required to avoid altering outer context: refer to https://github.com/autogluon/autogluon/issues/2688
            self._post_generators.append(
                RenameFeatureGenerator(name_prefix=name_prefix, name_suffix=name_suffix, inplace=False)
            )

        if self._post_generators:
            if not self.get_tags().get("allow_post_generators", True):
                raise AssertionError(
                    f"{self.__class__.__name__} is not allowed to have post_generators, "
                    f"but found: {[generator.__class__.__name__ for generator in self._post_generators]}"
                )

        self.pre_enforce_types = pre_enforce_types
        self._pre_astype_generator = None
        self.pre_drop_useless = pre_drop_useless
        self.reset_index = reset_index
        self.column_names_as_str = column_names_as_str
        self._useless_features_in: list = None

        self._is_updated_name = False  # If feature names have been altered by name_prefix or name_suffix

        self.log_prefix = log_prefix
        self.verbosity = verbosity

        self.fit_time = None

    def fit(self, X: DataFrame, **kwargs):
        """
        Fit generator to the provided data.
        Because of how the generators track output features and types, it is generally required that the data be transformed during fit, so the fit
        function is rarely useful to implement beyond a simple call to fit_transform.

        Parameters
        ----------
        X : DataFrame
            Input data used to fit the generator.
        **kwargs
            Any additional arguments that a particular generator implementation could use.
            See fit_transform method for common kwargs values.
        """
        self.fit_transform(X, **kwargs)

    def fit_transform(
            self, X: DataFrame, y: Series = None, feature_metadata_in: FeatureMetadata = None, **kwargs
    ) -> DataFrame:
        """
        Fit generator to the provided data and return the transformed version of the data as if fit and transform were called sequentially with the same data.
        This is generally more efficient than calling fit and transform separately and can be up to twice as fast if the fit process requires transformation
        of the data.
        This cannot be called after the generator has been fit, and will result in an AssertionError.

        Parameters
        ----------
        X : DataFrame
            Input data used to fit the generator.
        y : Series, optional
            Input data's labels used to fit the generator. Most generators do not utilize labels.
            y.index must be equal to X.index to avoid misalignment.
        feature_metadata_in : FeatureMetadata, optional
            Identical to providing feature_metadata_in during generator initialization. Ignored if self.feature_metadata_in is already specified.
            If neither are set, feature_metadata_in will be inferred from the _infer_feature_metadata_in method.
        **kwargs
            Any additional arguments that a particular generator implementation could use. Passed to _fit_transform and _fit_generators methods.

        Returns
        -------
        X_out : DataFrame object which is the transformed version of the input data X.

        """
        start_time = time.time()
        self._log(20, f"Fitting {self.__class__.__name__}...")
        if self._is_fit:
            raise AssertionError(f"{self.__class__.__name__} is already fit.")
        self._pre_fit_validate(X=X, y=y, feature_metadata_in=feature_metadata_in, **kwargs)

        if self.reset_index:
            X_index = copy.deepcopy(X.index)
            # TODO: Theoretically inplace=True avoids data copy, but can lead to altering of original DataFrame outside of method context.
            X = X.reset_index(drop=True)
            if y is not None and isinstance(y, Series):
                y = y.reset_index(drop=True)  # TODO: this assumes y and X had matching indices prior
        else:
            X_index = None
        if self.column_names_as_str:
            columns_orig = list(X.columns)
            X.columns = X.columns.astype(str)  # Ensure all column names are strings
            columns_new = list(X.columns)
            if columns_orig != columns_new:
                rename_map = {orig: new for orig, new in zip(columns_orig, columns_new)}
                if feature_metadata_in is not None:
                    feature_metadata_in.rename_features(rename_map=rename_map)
                self._rename_features_in(rename_map)
            else:
                self.column_names_as_str = False  # Columns were already string, so don't do conversion. Better to error if they change types at inference.
        self._ensure_no_duplicate_column_names(X=X)
        self._infer_features_in_full(X=X, feature_metadata_in=feature_metadata_in)
        if self.pre_drop_useless:
            self._useless_features_in = self._get_useless_features(X, columns_to_check=self.features_in)
            if self._useless_features_in:
                self._remove_features_in(self._useless_features_in)
        if self.pre_enforce_types:
            from .astype import AsTypeFeatureGenerator

            self._pre_astype_generator = AsTypeFeatureGenerator(
                features_in=self.features_in,
                feature_metadata_in=self.feature_metadata_in,
                log_prefix=self.log_prefix + "\t",
            )
            self._pre_astype_generator.fit(X)

        # TODO: Add option to return feature_metadata instead to avoid data copy
        #  If so, consider adding validation step to check that X_out matches the feature metadata, error/warning if not
        X_out, type_family_groups_special = self._fit_transform(X[self.features_in], y=y, **kwargs)
        self.features_in = list(X_out.columns)
        type_map_raw = get_type_map_raw(X_out)
        self._feature_metadata_before_post = FeatureMetadata(
            type_map_raw=type_map_raw, type_group_map_special=type_family_groups_special
        )
        if self._post_generators:
            X_out, self.feature_metadata, self._post_generators = self._fit_generators(
                X=X_out,
                y=y,
                feature_metadata=self._feature_metadata_before_post,
                generators=self._post_generators,
                **kwargs,
            )
        else:
            self.feature_metadata = self._feature_metadata_before_post
        type_map_real = get_type_map_real(X_out)
        self.features_out = list(X_out.columns)
        self.feature_metadata_real = FeatureMetadata(
            type_map_raw=type_map_real, type_group_map_special=self.feature_metadata.get_type_group_map_raw()
        )

        self._post_fit_cleanup()
        if self.reset_index:
            X_out.index = X_index
        self._is_fit = True
        end_time = time.time()
        self.fit_time = end_time - start_time
        if self.verbosity >= 3:
            self.print_feature_metadata_info(log_level=20)
            self.print_generator_info(log_level=20)
        elif self.verbosity == 2:
            self.print_feature_metadata_info(log_level=15)
            self.print_generator_info(log_level=15)
        return X_out

    def transform(self, X: DataFrame) -> DataFrame:
        """
        Transforms input data into the output data format.
        Will raise an AssertionError if called before the generator has been fit using fit or fit_transform methods.

        Parameters
        ----------
        X : DataFrame
            Input data to be transformed by the generator.
            Input data must contain all features in features_in, and should have the same dtypes as in the data provided to fit.
            Extra columns present in X that are not in features_in will be ignored and not affect the output.

        Returns
        -------
        X_out : DataFrame object which is the transformed version of the input data X.
        """
        if not self._is_fit:
            raise AssertionError(f"{self.__class__.__name__} is not fit.")
        if self.reset_index:
            X_index = copy.deepcopy(X.index)
            # TODO: Theoretically inplace=True avoids data copy, but can lead to altering of original DataFrame outside of method context.
            X = X.reset_index(drop=True)
        else:
            X_index = None
        if self.column_names_as_str:
            X.columns = X.columns.astype(str)  # Ensure all column names are strings
        try:
            if list(X.columns) != self.features_in:
                # It comes at a cost when making a copy of the DataFrame,
                # therefore, try avoid copying by checking the expected features first.
                X = X[self.features_in]
        except KeyError:
            missing_cols = []
            for col in self.features_in:
                if col not in X.columns:
                    missing_cols.append(col)
            raise KeyError(
                f"{len(missing_cols)} required columns are missing from the provided dataset to transform using {self.__class__.__name__}. "
                f"{len(missing_cols)} missing columns: {missing_cols} | "
                f"{len(list(X.columns))} available columns: {list(X.columns)}"
            )
        if self._pre_astype_generator:
            X = self._pre_astype_generator.transform(X)
        X_out = self._transform(X)
        if self._post_generators:
            X_out = self._transform_generators(X=X_out, generators=self._post_generators)
        if self.reset_index:
            X_out.index = X_index
        return X_out

    def _fit_transform(self, X: DataFrame, y: Series, **kwargs) -> (DataFrame, dict):
        """
        Performs the inner fit_transform logic that is non-generic (specific to the generator implementation).
        When creating a new generator class, this should be implemented.
        At the point this method is called, self.features_in and self.features_metadata_in will be set, and can be accessed and altered freely.

        Parameters
        ----------
        X : DataFrame
            Input data used to fit the generator.
            This data will have already been limited to only the columns present in self.features_in.
            This data may have been altered by the fit_transform method prior to entering _fit_transform in a variety of ways, but self.features_in and
            self.features_metadata_in will correctly correspond to X at this point in the generator's fit process.
        y : Series, optional
            Input data's labels used to fit the generator. Most generators do not utilize labels.
            y.index is always equal to X.index.
        **kwargs
            Any additional arguments that a particular generator implementation could use. Received from the fit_transform method.

        Returns
        -------
        (X_out : DataFrame, type_group_map_special : dict)
            X_out is the transformed version of the input data X
            type_group_map_special is the type_group_map_special value of X_out's intended FeatureMetadata object.
                If special types are not relevant to the generator, this can simply be dict()
                If the input and output features are identical in name and type, it may be valid to return self.feature_metadata_in.type_group_map_special
                to maintain any pre-existing special type information.
                Refer to existing generator implementations for guidance on setting the dict output of _fit_transform.

        """
        raise NotImplementedError

    def _transform(self, X: DataFrame) -> DataFrame:
        """
        Performs the inner transform logic that is non-generic (specific to the generator implementation).
        When creating a new generator class, this should be implemented.
        At the point this method is called, self.features_in and self.features_metadata_in will be set, and can be accessed freely.

        Parameters
        ----------
        X : DataFrame
            Input data to be transformed by the generator.
            This data will have already been limited to only the columns present in self.features_in.
            This data may have been altered by the transform method prior to entering _transform in a variety of ways, but self.features_in and
            self.features_metadata_in will correctly correspond to X at this point in the generator's transform process.

        Returns
        -------
        X_out : DataFrame object which is the transformed version of the input data X.
        """
        raise NotImplementedError

    def _infer_features_in_full(self, X: DataFrame, feature_metadata_in: FeatureMetadata = None):
        """
        Infers all input related feature information of X.
        This can be extended when additional input information is desired beyond feature_metadata_in and features_in.
            For example, AsTypeFeatureGenerator extends this method to also compute the exact raw feature types of the input for later use.
        After this method returns, self.features_in and self.feature_metadata_in will be set to proper values.
        This method is called by fit_transform prior to calling _fit_transform.

        Parameters
        ----------
        X : DataFrame
            Input data used to fit the generator.
        feature_metadata_in : FeatureMetadata, optional
            If passed, then self.feature_metadata_in will be set to feature_metadata_in assuming self.feature_metadata_in was None prior.
            If both are None, then self.feature_metadata_in is inferred through _infer_feature_metadata_in(X)
        """
        if self.feature_metadata_in is None:
            self.feature_metadata_in = feature_metadata_in
        elif feature_metadata_in is not None:
            self._log(
                30,
                "\tWarning: feature_metadata_in passed as input to fit_transform, but self.feature_metadata_in was already set. "
                "Ignoring feature_metadata_in.",
            )
        if self.feature_metadata_in is None:
            self._log(
                20,
                "\tInferring data type of each feature based on column values. Set feature_metadata_in to manually specify special "
                "dtypes of the features.",
            )
            self.feature_metadata_in = self._infer_feature_metadata_in(X=X)
        if self.features_in is None:
            self.features_in = self._infer_features_in(X=X)
            self.features_in = [feature for feature in self.features_in if feature in X.columns]
        self.feature_metadata_in = self.feature_metadata_in.keep_features(features=self.features_in)

    # TODO: Find way to increase flexibility here, possibly through init args
    def _infer_features_in(self, X: DataFrame) -> list:
        """
        Infers the features_in of X.
        This is used if features_in was not provided by the user prior to fit.
        This can be overwritten in a new generator to use new infer logic.
        self.feature_metadata_in is available at the time this method is called.

        Parameters
        ----------
        X : DataFrame
            Input data used to fit the generator.

        Returns
        -------
        feature_in : list of str feature names inferred from X.
        """
        return self.feature_metadata_in.get_features(**self._infer_features_in_args)

    # TODO: Use code from problem type detection for column types. Ints/Floats could be Categorical through this method. Maybe try both?
    @staticmethod
    def _infer_feature_metadata_in(X: DataFrame) -> FeatureMetadata:
        """
        Infers the feature_metadata_in of X.
        This is used if feature_metadata_in was not provided by the user prior to fit.
        This can be overwritten in a new generator to use new infer logic, but it is preferred to keep the default logic for consistency with other generators.

        Parameters
        ----------
        X : DataFrame
            Input data used to fit the generator.

        Returns
        -------
        feature_metadata_in : FeatureMetadata object inferred from X.
        """
        type_map_raw = get_type_map_raw(X)
        type_group_map_special = get_type_group_map_special(X)
        return FeatureMetadata(type_map_raw=type_map_raw, type_group_map_special=type_group_map_special)

    @staticmethod
    def get_default_infer_features_in_args() -> dict:
        raise NotImplementedError

    def _fit_generators(
            self, X, y, feature_metadata, generators: list, **kwargs
    ) -> (DataFrame, FeatureMetadata, list):
        """
        Fit a list of AbstractFeatureGenerator objects in sequence, with the output of generators[i] fed as the input to generators[i+1]
        This is called to sequentially fit self._post_generators generators on the output of _fit_transform to obtain the final output of the generator.
        This should not be overwritten by implementations of AbstractFeatureGenerator.
        """
        for generator in generators:
            generator.verbosity = min(self.verbosity, generator.verbosity)
            generator.set_log_prefix(log_prefix=self.log_prefix + "\t", prepend=True)
            X = generator.fit_transform(X=X, y=y, feature_metadata_in=feature_metadata, **kwargs)
            feature_metadata = generator.feature_metadata
        return X, feature_metadata, generators

    @staticmethod
    def _transform_generators(X, generators: list) -> DataFrame:
        """
        Transforms X through a list of AbstractFeatureGenerator objects in sequence, with the output of generators[i] fed as the input to generators[i+1]
        This is called to sequentially transform self._post_generators generators on the output of _transform to obtain the final output of the generator.
        This should not be overwritten by implementations of AbstractFeatureGenerator.
        """
        for generator in generators:
            X = generator.transform(X=X)
        return X

    def _remove_features_in(self, features: list):
        """
        Removes features from all relevant objects which represent the content of the input data or how the input features are used.
        For example, DropDuplicatesFeatureGenerator calls this method during _fit_transform with the list of duplicate features.
            This allows DropDuplicatesFeatureGenerator's _transform method to simply return X, as the duplicate features are already dropped in the transform
            method due to not being in self.features_in.

        Parameters
        ----------
        features : list of str
            List of feature names to remove from the expected input.
        """
        if features:
            if self._feature_metadata_before_post:
                feature_links_chain = self.get_feature_links_chain()
                for feature in features:
                    feature_links_chain[0].pop(feature)
                features_to_keep = set()
                for features_out in feature_links_chain[0].values():
                    features_to_keep = features_to_keep.union(features_out)
                self._feature_metadata_before_post = self._feature_metadata_before_post.keep_features(features_to_keep)

            self.feature_metadata_in = self.feature_metadata_in.remove_features(features=features)
            features_in_new = set(self.feature_metadata_in.get_features())
            self.features_in = [f for f in self.features_in if f in features_in_new]
            if self._pre_astype_generator:
                self._pre_astype_generator._remove_features_out(features)

    # TODO: Ensure arbitrary feature removal does not result in inconsistencies (add unit test)
    def _remove_features_out(self, features: list):
        """
        Removes features from the output data.
        This is used for cleaning complex pipelines of unnecessary operations after fitting a sequence of generators.
        Implementations of AbstractFeatureGenerator should not need to alter this method.

        Parameters
        ----------
        features : list of str
            List of feature names to remove from the output of self.transform().
        """
        feature_links_chain = self.get_feature_links_chain()
        if features:
            self.feature_metadata = self.feature_metadata.remove_features(features=features)
            self.feature_metadata_real = self.feature_metadata_real.remove_features(features=features)
            self.features_out = self.feature_metadata.get_features()
            feature_links_chain[-1] = {
                feature_in: [feature_out for feature_out in features_out if feature_out not in features]
                for feature_in, features_out in feature_links_chain[-1].items()
            }
        self._remove_unused_features(feature_links_chain=feature_links_chain)

    def _remove_unused_features(self, feature_links_chain):
        unused_features = self._get_unused_features(feature_links_chain=feature_links_chain)
        self._remove_features_in(features=unused_features[0])
        for i, generator in enumerate(self._post_generators):
            for feature in unused_features[i + 1]:
                if feature in feature_links_chain[i + 1]:
                    feature_links_chain[i + 1].pop(feature)
            generated_features = set()
            for feature_in in feature_links_chain[i + 1]:
                generated_features = generated_features.union(feature_links_chain[i + 1][feature_in])
            features_out_to_remove = [
                feature for feature in generator.features_out if feature not in generated_features
            ]
            generator._remove_features_out(features_out_to_remove)

    def _rename_features_in(self, column_rename_map: dict):
        if self.feature_metadata_in is not None:
            self.feature_metadata_in = self.feature_metadata_in.rename_features(column_rename_map)
        if self.features_in is not None:
            self.features_in = [column_rename_map.get(col, col) for col in self.features_in]

    def _pre_fit_validate(self, X: DataFrame, y: Series, **kwargs):
        """
        Any data validation checks prior to fitting the data should be done here.
        """
        if y is not None and isinstance(y, Series):
            if list(y.index) != list(X.index):
                raise AssertionError(
                    f"y.index and X.index must be equal when fitting {self.__class__.__name__}, but they differ."
                )

    def _post_fit_cleanup(self):
        """
        Any cleanup operations after all metadata objects have been constructed, but prior to feature renaming, should be done here.
        This includes removing keys from internal lists and dictionaries of features which have been removed, and deletion of any temp variables.
        """
        pass

    def _ensure_no_duplicate_column_names(self, X: DataFrame):
        if len(X.columns) != len(set(X.columns)):
            count_dict = defaultdict(int)
            invalid_columns = []
            for column in list(X.columns):
                count_dict[column] += 1
            for column in count_dict:
                if count_dict[column] > 1:
                    invalid_columns.append(column)
            raise AssertionError(
                f"Columns appear multiple times in X. Columns must be unique. Invalid columns: {invalid_columns}"
            )

    # TODO: Move to a generator
    @staticmethod
    def _get_useless_features(X: DataFrame, columns_to_check: List[str] = None) -> list:
        useless_features = []
        if columns_to_check is None:
            columns_to_check = list(X.columns)
        for column in columns_to_check:
            if is_useless_feature(X[column]):
                useless_features.append(column)
        return useless_features

    # TODO: Consider adding _log and verbosity methods to mixin
    def set_log_prefix(self, log_prefix, prepend=False):
        if prepend:
            self.log_prefix = log_prefix + self.log_prefix
        else:
            self.log_prefix = log_prefix

    def set_verbosity(self, verbosity: int):
        self.verbosity = verbosity

    def _log(self, level, msg, log_prefix=None, verb_min=None):
        if self.verbosity == 0:
            return
        if verb_min is None or self.verbosity >= verb_min:
            if log_prefix is None:
                log_prefix = self.log_prefix
            logger.log(level, f"{log_prefix}{msg}")

    def is_fit(self):
        return self._is_fit

    # TODO: Handle cases where self.features_in or self.feature_metadata_in was already set at init.
    def is_valid_metadata_in(self, feature_metadata_in: FeatureMetadata):
        """
        True if input data with feature metadata of feature_metadata_in could result in non-empty output.
            This is dictated by `feature_metadata_in.get_features(**self._infer_features_in_args)` not being empty.
        False if the features represented in feature_metadata_in do not contain any usable types for the generator.
            For example, if only numeric features are passed as input to TextSpecialFeatureGenerator which requires text input features, this will return False.
            However, if both numeric and text features are passed, this will return True since the text features would be valid input (the numeric features
            would simply be dropped).
        """
        features_in = feature_metadata_in.get_features(**self._infer_features_in_args)
        if features_in:
            return True
        else:
            return False

    def get_feature_links(self) -> Dict[str, List[str]]:
        """Returns feature links including all pre and post generators."""
        return self._get_feature_links_from_chain(self.get_feature_links_chain())

    def _get_feature_links(self, features_in: List[str], features_out: List[str]) -> Dict[str, List[str]]:
        """Returns feature links ignoring all pre and post generators."""
        feature_links = {}
        if self.get_tags().get("feature_interactions", True):
            for feature_in in features_in:
                feature_links[feature_in] = features_out
        else:
            for feat_old, feat_new in zip(features_in, features_out):
                feature_links[feat_old] = feature_links.get(feat_old, []) + [feat_new]
        return feature_links

    def get_feature_links_chain(self) -> List[Dict[str, List[str]]]:
        """Get the feature dependence chain between this generator and all of its post generators."""
        features_out_internal = self._feature_metadata_before_post.get_features()

        generators = [self] + self._post_generators
        features_in_list = [self.features_in] + [generator.features_in for generator in self._post_generators]
        features_out_list = [features_out_internal] + [generator.features_out for generator in self._post_generators]

        feature_links_chain = []
        for i in range(len(features_in_list)):
            generator = generators[i]
            features_in = features_in_list[i]
            features_out = features_out_list[i]
            feature_chain = generator._get_feature_links(features_in=features_in, features_out=features_out)
            feature_links_chain.append(feature_chain)
        return feature_links_chain

    @staticmethod
    def _get_feature_links_from_chain(feature_links_chain: List[Dict[str, List[str]]]) -> Dict[str, List[str]]:
        """Get the final input and output feature links by travelling the feature link chain"""
        features_out = []
        for val in feature_links_chain[-1].values():
            if val not in features_out:
                features_out.append(val)
        features_in = list(feature_links_chain[0].keys())
        feature_links = feature_links_chain[0]
        for i in range(1, len(feature_links_chain)):
            feature_links_new = {}
            for feature in features_in:
                feature_links_new[feature] = set()
                for feature_out in feature_links[feature]:
                    feature_links_new[feature] = feature_links_new[feature].union(
                        feature_links_chain[i].get(feature_out, [])
                    )
                feature_links_new[feature] = list(feature_links_new[feature])
            feature_links = feature_links_new
        return feature_links

    def _get_unused_features(self, feature_links_chain: List[Dict[str, List[str]]]):
        features_in_list = [self.features_in]
        if self._post_generators:
            for i in range(len(self._post_generators)):
                if i == 0:
                    features_in = self._feature_metadata_before_post.get_features()
                else:
                    features_in = self._post_generators[i - 1].features_out
                features_in_list.append(features_in)
        return self._get_unused_features_generic(
            feature_links_chain=feature_links_chain, features_in_list=features_in_list
        )

    # TODO: Unit test this
    @staticmethod
    def _get_unused_features_generic(
            feature_links_chain: List[Dict[str, List[str]]], features_in_list: List[List[str]]
    ) -> List[List[str]]:
        unused_features = []
        unused_features_by_stage = []
        for i, chain in enumerate(reversed(feature_links_chain)):
            stage = len(feature_links_chain) - i
            used_features = set()
            for key in chain.keys():
                new_val = [val for val in chain[key] if val not in unused_features]
                if new_val:
                    used_features.add(key)
            features_in = features_in_list[stage - 1]
            unused_features = []
            for feature in features_in:
                if feature not in used_features:
                    unused_features.append(feature)
            unused_features_by_stage.append(unused_features)
        unused_features_by_stage = list(reversed(unused_features_by_stage))
        return unused_features_by_stage

    def print_generator_info(self, log_level: int = 20):
        """
        Outputs detailed logs of the generator, such as the fit runtime.

        Parameters
        ----------
        log_level : int, default 20
            Log level of the logging statements.
        """
        if self.fit_time:
            self._log(log_level, f"\t{round(self.fit_time, 1)}s = Fit runtime")
            self._log(
                log_level,
                f"\t{len(self.features_in)} features in original data used to generate {len(self.features_out)} features in processed data.",
            )

    def print_feature_metadata_info(self, log_level: int = 20):
        """
        Outputs detailed logs of a fit feature generator including the input and output FeatureMetadata objects' feature types.

        Parameters
        ----------
        log_level : int, default 20
            Log level of the logging statements.
        """
        self._log(log_level, "\tTypes of features in original data (raw dtype, special dtypes):")
        self.feature_metadata_in.print_feature_metadata_full(self.log_prefix + "\t\t", log_level=log_level)
        if self.feature_metadata_real:
            self._log(log_level - 5, "\tTypes of features in processed data (exact raw dtype, raw dtype):")
            self.feature_metadata_real.print_feature_metadata_full(
                self.log_prefix + "\t\t", print_only_one_special=True, log_level=log_level - 5
            )
        self._log(log_level, "\tTypes of features in processed data (raw dtype, special dtypes):")
        self.feature_metadata.print_feature_metadata_full(self.log_prefix + "\t\t", log_level=log_level)

    def save(self, path: str):
        save_pkl.save(path=path, object=self)

    def _more_tags(self) -> dict:
        """
        Special values to enable advanced functionality.

        Tags
        ----
        feature_interactions : bool, default True
            If True, then treat all features_out as if they depend on all features_in.
            If False, then treat each features_out as if it was generated by a 1:1 mapping (no feature interactions).
                This enables advanced functionality regarding automated feature pruning, but is only valid for generators which only transform each feature
                and do not perform interactions.
        allow_post_generators : bool, default True
            If False, will raise an AssertionError if post_generators is specified during init.
                This is reserved for very simple generators where including post_generators would not be sensible, such as in RenameFeatureGenerator.
        """
        return {}

    def get_tags(self) -> dict:
        """Gets the tags for this generator."""
        collected_tags = {}
        for base_class in reversed(inspect.getmro(self.__class__)):
            if hasattr(base_class, "_more_tags"):
                # need the if because mixins might not have _more_tags
                # but might do redundant work in estimators
                # (i.e. calling more tags on BaseEstimator multiple times)
                more_tags = base_class._more_tags(self)
                collected_tags.update(more_tags)
        return collected_tags


class AbstractFeatureSelector:
    """
    Abstract feature selector implementation from which all AutoGluon feature selectors inherit.
    The purpose of a feature selector is to transform data from one form to another in a stateful manner.
    First, the selector is initialized with various arguments that dictate the way features are generated.
    Then, the selector is fit through either the `.fit()` or `.fit_transform()` methods using training data typically in pandas DataFrame format.
    Finally, the selector can transform new data with the same initial format as the training data through the `.transform()` method.

    Parameters
    ----------
    features_in : list, default None
        List of feature names the selector will expect and use in the fit and transform methods.
        Any feature in an incoming DataFrame that is not present in features_in is dropped and will not influence the transformation logic.
        If None, infer during fit from the _infer_features_in method.
        Equivalent to feature_metadata_in.get_features() post-fit.
    feature_metadata_in : :class:`autogluon.common.features.feature_metadata.FeatureMetadata`, default None
        :class:`FeatureMetadata` object corresponding to the training data input features.
        If None, infer during fit from the _infer_feature_metadata_in method.
        Any features not present in features_in (if provided) will be removed from feature_metadata_in.
    post_selectors : list of FeatureSelectors, default None
        FeatureSelectors which will fit and transform sequentially after this object's transformation logic,
        feeding their output into the next selector's input.
        The output of the final FeatureSelector will be the used as the transformed output.
    pre_enforce_types : bool, default False
        If True, the exact raw types (int64, float32, etc.) of the training data will be enforced on future data,
        either converting the types to the training types or raising an exception if unable.
        This is important to set to True on the outer feature selector in a feature generation pipeline to ensure
        incorrect dtypes are not passed downstream, but is often redundant when used on inner feature selectors inside a pipeline.
    pre_drop_useless : bool, default False
        If True, features_in will be pruned at fit time of features containing only a single unique value across all rows.
    post_drop_duplicates : bool, default False
        If True, a :class:`DropDuplicatesFeatureGenerator` will be appended to post_selectors.
        This feature selector will drop any duplicate features found in the data, keeping only one feature within any duplicate feature sets.
        Warning: For large datasets with many features, this may be very computationally expensive or even computationally infeasible.
    reset_index : bool, default False
        If True, for the duration of fit and transform, the input data's index is reset to be monotonically increasing from 0 to N-1 for a dataset of N rows.
        At the end of fit and transform, the original index is re-applied to the output data.
        This is important to set to True on the outer feature selector in a feature generation pipeline to ensure that a non-default
        index does not cause corruption of the inner feature generation if any inner feature selector does not properly handle non-default indices.
        This index reset is also applied to the y label data if provided during fit.
    column_names_as_str : bool, default True
        If True, the column names of the input data are converted to string if they were not already.
        This solves any issues related to downstream FeatureSelectors and models which cannot handle integer column names, and allows
        column name prefix and suffix operations to avoid errors.
        Note that for performance purposes, column names are only converted at transform time if they were not strings at fit time.
        Ensure consistent column names as input to avoid errors.
    name_prefix : str, default None
        Name prefix to add to all output feature names.
    name_suffix : str, default None
        Name suffix to add to all output feature names.
    infer_features_in_args : dict, default None
        Used as the kwargs input to FeatureMetadata.get_features(**kwargs) when inferring self.features_in.
        This is merged with the output dictionary of self.get_default_infer_features_in_args() depending on the value of infer_features_in_args_strategy.
        Only used when features_in is None.
        If None, then self.get_default_infer_features_in_args() is used directly.
        Refer to FeatureMetadata.get_features documentation for a full description of valid keys.
        Note: This is advanced functionality that is not necessary for most situations.
    infer_features_in_args_strategy : str, default 'overwrite'
        Determines how infer_features_in_args and self.get_default_infer_features_in_args() are combined to result in self._infer_features_in_args
        which dictates the features_in inference logic.
        If 'overwrite': infer_features_in_args is used exclusively and self.get_default_infer_features_in_args() is ignored.
        If 'update': self.get_default_infer_features_in_args() is dictionary updated by infer_features_in_args.
        If infer_features_in_args is None, this is ignored.
    banned_feature_special_types : List[str], default None
        List of feature special types to additionally exclude from input. Will update self.get_default_infer_features_in_args().
    log_prefix : str, default ''
        Prefix string added to all logging statements made by the selector.
    verbosity : int, default 2
        Controls the verbosity of logging.
        0 will silence logs, 1 will only log warnings, 2 will log info level information, and 3 will log info level information and provide detailed
        feature type input and output information.
        Logging is still controlled by the global logger configuration, and therefore a verbosity of 3 does not guarantee that logs will be output.

    Attributes
    ----------
    features_in : list of str
        List of feature names the selector will expect and use in the fit and transform methods.
        Equivalent to feature_metadata_in.get_features() post-fit.
    features_out : list of str
        List of feature names present in the output of fit_transform and transform methods.
        Equivalent to feature_metadata.get_features() post-fit.
    feature_metadata_in : FeatureMetadata
        The FeatureMetadata of data pre-transformation (data used as input to fit and transform methods).
    feature_metadata : FeatureMetadata
        The FeatureMetadata of data post-transformation (data outputted by fit_transform and transform methods).
    feature_metadata_real : FeatureMetadata
        The FeatureMetadata of data post-transformation consisting of the exact dtypes as opposed to the grouped raw dtypes found in feature_metadata_in,
        with grouped raw dtypes substituting for the special dtypes.
        This is only used in the print_feature_metadata_info method and is intended for introspection. It can be safely set to None to reduce memory and
        disk usage post-fit.
    """
    model_file_name = "model.pkl"
    model_info_name = "info.pkl"
    model_info_json_name = "info.json"

    seed_name: str | None = None
    seed_name_alt: list[str] = []
    default_random_seed: int | None = 0

    def __init__(
            self,
            features_in: list = None,
            feature_metadata_in: FeatureMetadata = None,
            post_selectors: list = None,
            pre_enforce_types=False,
            pre_drop_useless=False,
            post_drop_duplicates=False,
            reset_index=False,
            column_names_as_str=True,
            name_prefix: str = None,
            name_suffix: str = None,
            infer_features_in_args: dict = None,
            infer_features_in_args_strategy="overwrite",
            banned_feature_special_types: List[str] = None,
            hyperparameters: dict | None = None,
            problem_type: str | None = None,
            name: str | None = None,
            path: str | None = None,
            time_limit: float = None,
            log_prefix="",
            verbosity=2,
    ):
        self.time_limit = time_limit
        self.features = None
        self.params = {}
        self.params_aux = {}
        if name is None:
            self.name = self.__class__.__name__
            logger.log(20, f"Warning: No name was specified for method, defaulting to {self.name}")
        else:
            self.name = name
        self.path_root = path
        if self.path_root is None:
            path_suffix = self.name
            # TODO: Would be ideal to not create dir, but still track that it is unique. However, this isn't possible to do without a global list of used dirs or using UUID.
            path_cur = setup_outputdir(path=None, create_dir=True, path_suffix=path_suffix)
            self.path_root = path_cur.rsplit(self.path_suffix, 1)[0]
            logger.log(20, f"Warning: No path was specified for model, defaulting to: {self.path_root}")
        self.path = self.create_contexts(os.path.join(self.path_root, self.path_suffix))
        self.problem_type = problem_type
        self._is_initialized = False
        self._user_params, self._user_params_aux = self._init_user_params(params=hyperparameters)
        self._is_fit_metadata_registered = False
        self._is_fit = False  # Whether the feature selector has been fit
        self.features_in = features_in  # Original features to use as input to feature generation
        self.features_out = None  # Final list of features after transformation
        self.feature_metadata_in: FeatureMetadata = (
            feature_metadata_in  # FeatureMetadata object based on the original input features.
        )

        # FeatureMetadata object based on the processed features. Pass to models to enable advanced functionality.
        self.feature_metadata: FeatureMetadata = None

        # TODO: Consider merging feature_metadata and feature_metadata_real, have FeatureMetadata contain exact dtypes, grouped raw dtypes,
        #  and special dtypes all at once.
        # FeatureMetadata object based on the processed features, containing the true raw dtype information (such as int32, float64, etc.).
        # Pass to models to enable advanced functionality.
        self._features_internal = None
        self.feature_metadata_real: FeatureMetadata = None
        self._feature_metadata_before_post = None  # FeatureMetadata directly prior to applying self._post_selectors.
        self._infer_features_in_args = self.get_default_infer_features_in_args()
        if infer_features_in_args is not None:
            if infer_features_in_args_strategy == "overwrite":
                self._infer_features_in_args = copy.deepcopy(infer_features_in_args)
            elif infer_features_in_args_strategy == "update":
                self._infer_features_in_args.update(infer_features_in_args)
            else:
                raise ValueError(
                    f"infer_features_in_args_strategy must be one of: {['overwrite', 'update']}, but was: '{infer_features_in_args_strategy}'"
                )
        if banned_feature_special_types:
            if "invalid_special_types" not in self._infer_features_in_args:
                self._infer_features_in_args["invalid_special_types"] = banned_feature_special_types
            else:
                for f in banned_feature_special_types:
                    if f not in self._infer_features_in_args["invalid_special_types"]:
                        self._infer_features_in_args["invalid_special_types"].append(f)

        if post_selectors is None:
            post_selectors = []
        elif not isinstance(post_selectors, list):
            post_selectors = [post_selectors]
        self._post_selectors: list = post_selectors
        if post_drop_duplicates:
            from .drop_duplicates import DropDuplicatesFeatureGenerator

            self._post_selectors.append(DropDuplicatesFeatureGenerator(post_drop_duplicates=False))
        if name_prefix or name_suffix:
            from .rename import RenameFeatureGenerator

            # inplace=False required to avoid altering outer context: refer to https://github.com/autogluon/autogluon/issues/2688
            self._post_selectors.append(
                RenameFeatureGenerator(name_prefix=name_prefix, name_suffix=name_suffix, inplace=False)
            )

        if self._post_selectors:
            if not self.get_tags().get("allow_post_selectors", True):
                raise AssertionError(
                    f"{self.__class__.__name__} is not allowed to have post_selectors, "
                    f"but found: {[selector.__class__.__name__ for selector in self._post_selectors]}"
                )

        self.pre_enforce_types = pre_enforce_types
        self._pre_astype_selector = None
        self.pre_drop_useless = pre_drop_useless
        self.reset_index = reset_index
        self.column_names_as_str = column_names_as_str
        self._useless_features_in: list = None

        self._is_updated_name = False  # If feature names have been altered by name_prefix or name_suffix

        self.log_prefix = log_prefix
        self.verbosity = verbosity

        self.fit_time = None

        self.random_seed: int | None | str = "NOTSET"

    @staticmethod
    def create_contexts(path_context: str) -> str:
        path = path_context
        return path

    def fit(self, X: DataFrame, n_max_features: int = None, **kwargs):
        """
        Fit selector to the provided data.
        Because of how the selectors track output features and types, it is generally required that the data be transformed during fit, so the fit
        function is rarely useful to implement beyond a simple call to fit_transform.

        Parameters
        ----------
        X : DataFrame
            Input data used to fit the selector.
        **kwargs
            Any additional arguments that a particular selector implementation could use.
            See fit_transform method for common kwargs values.
        """
        self.fit_transform(X, n_max_features, **kwargs)

    def fit_transform(
            self,
            X: DataFrame,
            y: Series,
            model=None,
            n_max_features: int = None,
            time_limit: float = 600,
            feature_metadata_in: FeatureMetadata = None,
            log_resources: bool = True,
            log_resources_prefix: str | None = None,
            **kwargs
    ) -> DataFrame:
        """
        Fit selector to the provided data and return the transformed version of the data as if fit and transform were called sequentially with the same data.
        This is generally more efficient than calling fit and transform separately and can be up to twice as fast if the fit process requires transformation
        of the data.
        This cannot be called after the selector has been fit, and will result in an AssertionError.

        Parameters
        ----------
        X : DataFrame
            Input data used to fit the selector.
        y : Series, optional
            Input data's labels used to fit the selector. Most selectors do not utilize labels.
            y.index must be equal to X.index to avoid misalignment.
        feature_metadata_in : FeatureMetadata, optional
            Identical to providing feature_metadata_in during selector initialization. Ignored if self.feature_metadata_in is already specified.
            If neither are set, feature_metadata_in will be inferred from the _infer_feature_metadata_in method.
        log_resources : bool, default = False
            If True, will log information about the number of CPUs, GPUs, and memory usage during fit.
        log_resources_prefix : str | None, default = None
            If specified, will be prepended to the log generated when `log_resources=True`.
        **kwargs
            Any additional arguments that a particular selector implementation could use. Passed to _fit_transform and _fit_selectors methods.

        Returns
        -------
        X_out : DataFrame object which is the transformed version of the input data X.

        """
        try:
            start_time = time.time()
            kwargs = self.initialize(time_limit=time_limit, **kwargs)
            kwargs["X"] = X
            self.model = model
            kwargs["start_time"] = start_time
            self._register_fit_metadata(**kwargs)
            self.validate_fit_resources(**kwargs)
            approx_mem_size_req, available_mem = self._validate_fit_memory_usage(**kwargs)
            if "time_limit" in kwargs and kwargs["time_limit"] is not None:
                time_start_fit = time.time()
                kwargs["time_limit"] -= time_start_fit - start_time
                if kwargs["time_limit"] <= 0:
                    logger.warning(
                        f'\tWarning: FeatureSelection Method has no time left to train... (Time Left = {kwargs["time_limit"]:.1f}s)')
                    raise TimeLimitExceeded
            self.validate_fit_args(**kwargs)
            if log_resources:
                num_cpus = kwargs.get("num_cpus", None)
                num_gpus = kwargs.get("num_gpus", None)
                approx_mem_size_req_gb = approx_mem_size_req / (1024 ** 3) if approx_mem_size_req is not None else None
                print("Approx. Required Memory in GB: " + str(approx_mem_size_req_gb))
                available_mem_gb = available_mem / (1024 ** 3) if available_mem is not None else None
                print("Available Memory in GB: " + str(available_mem_gb))

                if log_resources_prefix is None:
                    log_resources_prefix = ""
                msg = f"\t{log_resources_prefix}Fitting with cpus={num_cpus}, gpus={num_gpus}"
                if approx_mem_size_req_gb is not None and available_mem_gb is not None:
                    msg_mem = f", mem={approx_mem_size_req_gb:.1f}/{available_mem_gb:.1f} GB"
                    msg += msg_mem
                logger.log(20, msg)

            self._log(20, f"Fitting {self.__class__.__name__}...")
            if self._is_fit:
                raise AssertionError(f"{self.__class__.__name__} is already fit.")
            kwargs.pop("X", None)
            self._pre_fit_validate(X=X, y=y, feature_metadata_in=feature_metadata_in, **kwargs)

            if self.reset_index:
                X_index = copy.deepcopy(X.index)
                # TODO: Theoretically inplace=True avoids data copy, but can lead to altering of original DataFrame outside of method context.
                X = X.reset_index(drop=True)
                if y is not None and isinstance(y, Series):
                    y = y.reset_index(drop=True)  # TODO: this assumes y and X had matching indices prior
            else:
                X_index = None
            if self.column_names_as_str:
                columns_orig = list(X.columns)
                X.columns = X.columns.astype(str)  # Ensure all column names are strings
                columns_new = list(X.columns)
                if columns_orig != columns_new:
                    rename_map = {orig: new for orig, new in zip(columns_orig, columns_new)}
                    if feature_metadata_in is not None:
                        feature_metadata_in.rename_features(rename_map=rename_map)
                    self._rename_features_in(rename_map)
                else:
                    self.column_names_as_str = False  # Columns were already string, so don't do conversion. Better to error if they change types at inference.
            self._ensure_no_duplicate_column_names(X=X)
            self._infer_features_in_full(X=X, feature_metadata_in=feature_metadata_in)
            if self.pre_drop_useless:
                self._useless_features_in = self._get_useless_features(X, columns_to_check=self.features_in)
                if self._useless_features_in:
                    self._remove_features_in(self._useless_features_in)
            if self.pre_enforce_types:
                from .astype import AsTypeFeatureSelector

                self._pre_astype_selector = AsTypeFeatureSelector(
                    features_in=self.features_in,
                    feature_metadata_in=self.feature_metadata_in,
                    log_prefix=self.log_prefix + "\t",
                )
                self._pre_astype_selector.fit(X)

            self.features_in = list(X.columns)
            X_out, type_family_groups_special = self._fit_transform(X=X[self.features_in], y=y, model=self.model,
                                                                    n_max_features=n_max_features, **kwargs)
        except TimeLimitExceeded:
            if n_max_features is None:
                X_out = X
                type_family_groups_special = {}
            else:
                X_out = X.sample(n=n_max_features, axis=1)
                type_family_groups_special = {}
            if self.feature_metadata_in is not None:
                self._feature_metadata_before_post = copy.deepcopy(self.feature_metadata_in)
                self.feature_metadata = copy.deepcopy(self.feature_metadata_in)
                self.features_in = list(X_out.columns)
            else:
                type_map_raw = get_type_map_raw(X_out)
                metadata = FeatureMetadata(type_map_raw=type_map_raw)
                self._feature_metadata_before_post = metadata
                self.feature_metadata = metadata
                self.features_in = list(X_out.columns)

        type_map_raw = get_type_map_raw(X_out)
        self._feature_metadata_before_post = FeatureMetadata(
            type_map_raw=type_map_raw, type_group_map_special=type_family_groups_special
        )
        if self._post_selectors:
            X_out, self.feature_metadata, self._post_selectors = self._fit_selectors(
                X=X_out,
                y=y,
                n_max_features=n_max_features,
                feature_metadata=self._feature_metadata_before_post,
                selectors=self._post_selectors,
                **kwargs,
            )
        else:
            self.feature_metadata = self._feature_metadata_before_post
        type_map_real = get_type_map_real(X_out)
        self.features_out = list(X_out.columns)
        self.feature_metadata_real = FeatureMetadata(
            type_map_raw=type_map_real, type_group_map_special=self.feature_metadata.get_type_group_map_raw()
        )

        self._post_fit_cleanup()
        if self.reset_index:
            X_out.index = X_index
        self._is_fit = True
        end_time = time.time()
        self.fit_time = end_time - start_time
        if self.verbosity >= 3:
            self.print_feature_metadata_info(log_level=20)
            self.print_selector_info(log_level=20)
        elif self.verbosity == 2:
            self.print_feature_metadata_info(log_level=15)
            self.print_selector_info(log_level=15)

        return X_out

    def transform(self, X: DataFrame) -> DataFrame:
        """
        Transforms input data into the output data format.
        Will raise an AssertionError if called before the selector has been fit using fit or fit_transform methods.

        Parameters
        ----------
        X : DataFrame
            Input data to be transformed by the selector.
            Input data must contain all features in features_in, and should have the same dtypes as in the data provided to fit.
            Extra columns present in X that are not in features_in will be ignored and not affect the output.

        Returns
        -------
        X_out : DataFrame object which is the transformed version of the input data X.
        """
        if not self._is_fit:
            raise AssertionError(f"{self.__class__.__name__} is not fit.")
        if self.reset_index:
            X_index = copy.deepcopy(X.index)
            # TODO: Theoretically inplace=True avoids data copy, but can lead to altering of original DataFrame outside of method context.
            X = X.reset_index(drop=True)
        else:
            X_index = None
        if self.column_names_as_str:
            X.columns = X.columns.astype(str)  # Ensure all column names are strings
        try:
            if list(X.columns) != self.features_in:
                # It comes at a cost when making a copy of the DataFrame,
                # therefore, try avoid copying by checking the expected features first.
                X = X[self.features_in]
        except KeyError:
            missing_cols = []
            for col in self.features_in:
                if col not in X.columns:
                    missing_cols.append(col)
            raise KeyError(
                f"{len(missing_cols)} required columns are missing from the provided dataset to transform using {self.__class__.__name__}. "
                f"{len(missing_cols)} missing columns: {missing_cols} | "
                f"{len(list(X.columns))} available columns: {list(X.columns)}"
            )
        if self._pre_astype_selector:
            X = self._pre_astype_selector.transform(X)
        X_out = self._transform(X)
        if self._post_selectors:
            X_out = self._transform_selectors(X=X_out, selectors=self._post_selectors)
        if self.reset_index:
            X_out.index = X_index
        return X_out

    def _fit_transform(self, X: DataFrame, y: Series, model, n_max_features: int, **kwargs) -> (DataFrame, dict):
        """
        Performs the inner fit_transform logic that is non-generic (specific to the selector implementation).
        When creating a new selector class, this should be implemented.
        At the point this method is called, self.features_in and self.features_metadata_in will be set, and can be accessed and altered freely.

        Parameters
        ----------
        X : DataFrame
            Input data used to fit the selector.
            This data will have already been limited to only the columns present in self.features_in.
            This data may have been altered by the fit_transform method prior to entering _fit_transform in a variety of ways, but self.features_in and
            self.features_metadata_in will correctly correspond to X at this point in the selector's fit process.
        y : Series, optional
            Input data's labels used to fit the selector. Most selectors do not utilize labels.
            y.index is always equal to X.index.
        **kwargs
            Any additional arguments that a particular selector implementation could use. Received from the fit_transform method.

        Returns
        -------
        (X_out : DataFrame, type_group_map_special : dict)
            X_out is the transformed version of the input data X
            type_group_map_special is the type_group_map_special value of X_out's intended FeatureMetadata object.
                If special types are not relevant to the selector, this can simply be dict()
                If the input and output features are identical in name and type, it may be valid to return self.feature_metadata_in.type_group_map_special
                to maintain any pre-existing special type information.
                Refer to existing selector implementations for guidance on setting the dict output of _fit_transform.

        """
        raise NotImplementedError

    def _transform(self, X: DataFrame) -> DataFrame:
        """
        Performs the inner transform logic that is non-generic (specific to the selector implementation).
        When creating a new selector class, this should be implemented.
        At the point this method is called, self.features_in and self.features_metadata_in will be set, and can be accessed freely.

        Parameters
        ----------
        X : DataFrame
            Input data to be transformed by the selector.
            This data will have already been limited to only the columns present in self.features_in.
            This data may have been altered by the transform method prior to entering _transform in a variety of ways, but self.features_in and
            self.features_metadata_in will correctly correspond to X at this point in the selector's transform process.

        Returns
        -------
        X_out : DataFrame object which is the transformed version of the input data X.
        """
        raise NotImplementedError

    def _infer_features_in_full(self, X: DataFrame, feature_metadata_in: FeatureMetadata = None):
        """
        Infers all input related feature information of X.
        This can be extended when additional input information is desired beyond feature_metadata_in and features_in.
            For example, AsTypeFeatureSelector extends this method to also compute the exact raw feature types of the input for later use.
        After this method returns, self.features_in and self.feature_metadata_in will be set to proper values.
        This method is called by fit_transform prior to calling _fit_transform.

        Parameters
        ----------
        X : DataFrame
            Input data used to fit the selector.
        feature_metadata_in : FeatureMetadata, optional
            If passed, then self.feature_metadata_in will be set to feature_metadata_in assuming self.feature_metadata_in was None prior.
            If both are None, then self.feature_metadata_in is inferred through _infer_feature_metadata_in(X)
        """
        if self.feature_metadata_in is None:
            self.feature_metadata_in = feature_metadata_in
        elif feature_metadata_in is not None:
            self._log(
                30,
                "\tWarning: feature_metadata_in passed as input to fit_transform, but self.feature_metadata_in was already set. "
                "Ignoring feature_metadata_in.",
            )
        if self.feature_metadata_in is None:
            self._log(
                20,
                "\tInferring data type of each feature based on column values. Set feature_metadata_in to manually specify special "
                "dtypes of the features.",
            )
            self.feature_metadata_in = self._infer_feature_metadata_in(X=X)
        if self.features_in is None:
            self.features_in = self._infer_features_in(X=X)
            self.features_in = [feature for feature in self.features_in if feature in X.columns]
        self.feature_metadata_in = self.feature_metadata_in.keep_features(features=self.features_in)

    # TODO: Find way to increase flexibility here, possibly through init args
    def _infer_features_in(self, X: DataFrame) -> list:
        """
        Infers the features_in of X.
        This is used if features_in was not provided by the user prior to fit.
        This can be overwritten in a new selector to use new infer logic.
        self.feature_metadata_in is available at the time this method is called.

        Parameters
        ----------
        X : DataFrame
            Input data used to fit the selector.

        Returns
        -------
        feature_in : list of str feature names inferred from X.
        """
        return self.feature_metadata_in.get_features(**self._infer_features_in_args)

    # TODO: Use code from problem type detection for column types. Ints/Floats could be Categorical through this method. Maybe try both?
    @staticmethod
    def _infer_feature_metadata_in(X: DataFrame) -> FeatureMetadata:
        """
        Infers the feature_metadata_in of X.
        This is used if feature_metadata_in was not provided by the user prior to fit.
        This can be overwritten in a new selector to use new infer logic, but it is preferred to keep the default logic for consistency with other selectors.

        Parameters
        ----------
        X : DataFrame
            Input data used to fit the selector.

        Returns
        -------
        feature_metadata_in : FeatureMetadata object inferred from X.
        """
        type_map_raw = get_type_map_raw(X)
        type_group_map_special = get_type_group_map_special(X)
        return FeatureMetadata(type_map_raw=type_map_raw, type_group_map_special=type_group_map_special)

    @staticmethod
    def get_default_infer_features_in_args() -> dict:
        raise NotImplementedError

    def _fit_selectors(
            self, X, y, n_max_features, feature_metadata, selectors: list, **kwargs
    ) -> (DataFrame, FeatureMetadata, list):
        """
        Fit a list of AbstractFeatureSelector objects in sequence, with the output of selectors[i] fed as the input to selectors[i+1]
        This is called to sequentially fit self._post_selectors selectors on the output of _fit_transform to obtain the final output of the selector.
        This should not be overwritten by implementations of AbstractFeatureSelector.
        """
        for selector in selectors:
            selector.verbosity = min(self.verbosity, selector.verbosity)
            selector.set_log_prefix(log_prefix=self.log_prefix + "\t", prepend=True)
            X = selector.fit_transform(X=X, y=y, n_max_features=n_max_features, feature_metadata_in=feature_metadata,
                                       **kwargs)
            feature_metadata = selector.feature_metadata
        return X, feature_metadata, selectors

    @staticmethod
    def _transform_selectors(X, selectors: list) -> DataFrame:
        """
        Transforms X through a list of AbstractFeatureSelector objects in sequence, with the output of selectors[i] fed as the input to selectors[i+1]
        This is called to sequentially transform self._post_selectors selectors on the output of _transform to obtain the final output of the selector.
        This should not be overwritten by implementations of AbstractFeatureSelector.
        """
        for selector in selectors:
            X = selector.transform(X=X)
        return X

    def _remove_features_in(self, features: list):
        """
        Removes features from all relevant objects which represent the content of the input data or how the input features are used.
        For example, DropDuplicatesFeatureGenerator calls this method during _fit_transform with the list of duplicate features.
            This allows DropDuplicatesFeatureGenerator's _transform method to simply return X, as the duplicate features are already dropped in the transform
            method due to not being in self.features_in.

        Parameters
        ----------
        features : list of str
            List of feature names to remove from the expected input.
        """
        if features:
            if self._feature_metadata_before_post:
                feature_links_chain = self.get_feature_links_chain()
                for feature in features:
                    feature_links_chain[0].pop(feature)
                features_to_keep = set()
                for features_out in feature_links_chain[0].values():
                    features_to_keep = features_to_keep.union(features_out)
                self._feature_metadata_before_post = self._feature_metadata_before_post.keep_features(features_to_keep)

            self.feature_metadata_in = self.feature_metadata_in.remove_features(features=features)
            features_in_new = set(self.feature_metadata_in.get_features())
            self.features_in = [f for f in self.features_in if f in features_in_new]
            if self._pre_astype_selector:
                self._pre_astype_selector._remove_features_out(features)

    # TODO: Ensure arbitrary feature removal does not result in inconsistencies (add unit test)
    def _remove_features_out(self, features: list):
        """
        Removes features from the output data.
        This is used for cleaning complex pipelines of unnecessary operations after fitting a sequence of selectors.
        Implementations of AbstractFeatureSelector should not need to alter this method.

        Parameters
        ----------
        features : list of str
            List of feature names to remove from the output of self.transform().
        """
        feature_links_chain = self.get_feature_links_chain()
        if features:
            self.feature_metadata = self.feature_metadata.remove_features(features=features)
            self.feature_metadata_real = self.feature_metadata_real.remove_features(features=features)
            self.features_out = self.feature_metadata.get_features()
            feature_links_chain[-1] = {
                feature_in: [feature_out for feature_out in features_out if feature_out not in features]
                for feature_in, features_out in feature_links_chain[-1].items()
            }
        self._remove_unused_features(feature_links_chain=feature_links_chain)

    def _remove_unused_features(self, feature_links_chain):
        unused_features = self._get_unused_features(feature_links_chain=feature_links_chain)
        self._remove_features_in(features=unused_features[0])
        for i, selector in enumerate(self._post_selectors):
            for feature in unused_features[i + 1]:
                if feature in feature_links_chain[i + 1]:
                    feature_links_chain[i + 1].pop(feature)
            generated_features = set()
            for feature_in in feature_links_chain[i + 1]:
                generated_features = generated_features.union(feature_links_chain[i + 1][feature_in])
            features_out_to_remove = [
                feature for feature in selector.features_out if feature not in generated_features
            ]
            selector._remove_features_out(features_out_to_remove)

    def _rename_features_in(self, column_rename_map: dict):
        if self.feature_metadata_in is not None:
            self.feature_metadata_in = self.feature_metadata_in.rename_features(column_rename_map)
        if self.features_in is not None:
            self.features_in = [column_rename_map.get(col, col) for col in self.features_in]

    def _pre_fit_validate(self, X: DataFrame, y: Series, **kwargs):
        """
        Any data validation checks prior to fitting the data should be done here.
        """
        if y is not None and isinstance(y, Series):
            if list(y.index) != list(X.index):
                raise AssertionError(
                    f"y.index and X.index must be equal when fitting {self.__class__.__name__}, but they differ."
                )

    def _post_fit_cleanup(self):
        """
        Any cleanup operations after all metadata objects have been constructed, but prior to feature renaming, should be done here.
        This includes removing keys from internal lists and dictionaries of features which have been removed, and deletion of any temp variables.
        """
        pass

    def _ensure_no_duplicate_column_names(self, X: DataFrame):
        if len(X.columns) != len(set(X.columns)):
            count_dict = defaultdict(int)
            invalid_columns = []
            for column in list(X.columns):
                count_dict[column] += 1
            for column in count_dict:
                if count_dict[column] > 1:
                    invalid_columns.append(column)
            raise AssertionError(
                f"Columns appear multiple times in X. Columns must be unique. Invalid columns: {invalid_columns}"
            )

    # TODO: Move to a selector
    @staticmethod
    def _get_useless_features(X: DataFrame, columns_to_check: List[str] = None) -> list:
        useless_features = []
        if columns_to_check is None:
            columns_to_check = list(X.columns)
        for column in columns_to_check:
            if is_useless_feature(X[column]):
                useless_features.append(column)
        return useless_features

    # TODO: Consider adding _log and verbosity methods to mixin
    def set_log_prefix(self, log_prefix, prepend=False):
        if prepend:
            self.log_prefix = log_prefix + self.log_prefix
        else:
            self.log_prefix = log_prefix

    def set_verbosity(self, verbosity: int):
        self.verbosity = verbosity

    def _log(self, level, msg, log_prefix=None, verb_min=None):
        if self.verbosity == 0:
            return
        if verb_min is None or self.verbosity >= verb_min:
            if log_prefix is None:
                log_prefix = self.log_prefix
            logger.log(level, f"{log_prefix}{msg}")

    def is_fit(self):
        return self._is_fit

    # TODO: Handle cases where self.features_in or self.feature_metadata_in was already set at init.
    def is_valid_metadata_in(self, feature_metadata_in: FeatureMetadata):
        """
        True if input data with feature metadata of feature_metadata_in could result in non-empty output.
            This is dictated by `feature_metadata_in.get_features(**self._infer_features_in_args)` not being empty.
        False if the features represented in feature_metadata_in do not contain any usable types for the selector.
            For example, if only numeric features are passed as input to TextSpecialFeatureGenerator which requires text input features, this will return False.
            However, if both numeric and text features are passed, this will return True since the text features would be valid input (the numeric features
            would simply be dropped).
        """
        features_in = feature_metadata_in.get_features(**self._infer_features_in_args)
        if features_in:
            return True
        else:
            return False

    def get_feature_links(self) -> Dict[str, List[str]]:
        """Returns feature links including all pre and post selectors."""
        return self._get_feature_links_from_chain(self.get_feature_links_chain())

    def _get_feature_links(self, features_in: List[str], features_out: List[str]) -> Dict[str, List[str]]:
        """Returns feature links ignoring all pre and post selectors."""
        feature_links = {}
        if self.get_tags().get("feature_interactions", True):
            for feature_in in features_in:
                feature_links[feature_in] = features_out
        else:
            for feat_old, feat_new in zip(features_in, features_out):
                feature_links[feat_old] = feature_links.get(feat_old, []) + [feat_new]
        return feature_links

    def get_feature_links_chain(self) -> List[Dict[str, List[str]]]:
        """Get the feature dependence chain between this selector and all of its post selectors."""
        features_out_internal = self._feature_metadata_before_post.get_features()

        selectors = [self] + self._post_selectors
        features_in_list = [self.features_in] + [selector.features_in for selector in self._post_selectors]
        features_out_list = [features_out_internal] + [selector.features_out for selector in self._post_selectors]

        feature_links_chain = []
        for i in range(len(features_in_list)):
            selector = selectors[i]
            features_in = features_in_list[i]
            features_out = features_out_list[i]
            feature_chain = selector._get_feature_links(features_in=features_in, features_out=features_out)
            feature_links_chain.append(feature_chain)
        return feature_links_chain

    @staticmethod
    def _get_feature_links_from_chain(feature_links_chain: List[Dict[str, List[str]]]) -> Dict[str, List[str]]:
        """Get the final input and output feature links by travelling the feature link chain"""
        features_out = []
        for val in feature_links_chain[-1].values():
            if val not in features_out:
                features_out.append(val)
        features_in = list(feature_links_chain[0].keys())
        feature_links = feature_links_chain[0]
        for i in range(1, len(feature_links_chain)):
            feature_links_new = {}
            for feature in features_in:
                feature_links_new[feature] = set()
                for feature_out in feature_links[feature]:
                    feature_links_new[feature] = feature_links_new[feature].union(
                        feature_links_chain[i].get(feature_out, [])
                    )
                feature_links_new[feature] = list(feature_links_new[feature])
            feature_links = feature_links_new
        return feature_links

    def _get_unused_features(self, feature_links_chain: List[Dict[str, List[str]]]):
        features_in_list = [self.features_in]
        if self._post_selectors:
            for i in range(len(self._post_selectors)):
                if i == 0:
                    features_in = self._feature_metadata_before_post.get_features()
                else:
                    features_in = self._post_selectors[i - 1].features_out
                features_in_list.append(features_in)
        return self._get_unused_features_generic(
            feature_links_chain=feature_links_chain, features_in_list=features_in_list
        )

    # TODO: Unit test this
    @staticmethod
    def _get_unused_features_generic(
            feature_links_chain: List[Dict[str, List[str]]], features_in_list: List[List[str]]
    ) -> List[List[str]]:
        unused_features = []
        unused_features_by_stage = []
        for i, chain in enumerate(reversed(feature_links_chain)):
            stage = len(feature_links_chain) - i
            used_features = set()
            for key in chain.keys():
                new_val = [val for val in chain[key] if val not in unused_features]
                if new_val:
                    used_features.add(key)
            features_in = features_in_list[stage - 1]
            unused_features = []
            for feature in features_in:
                if feature not in used_features:
                    unused_features.append(feature)
            unused_features_by_stage.append(unused_features)
        unused_features_by_stage = list(reversed(unused_features_by_stage))
        return unused_features_by_stage

    def print_selector_info(self, log_level: int = 20):
        """
        Outputs detailed logs of the selector, such as the fit runtime.

        Parameters
        ----------
        log_level : int, default 20
            Log level of the logging statements.
        """
        if self.fit_time:
            self._log(log_level, f"\t{round(self.fit_time, 1)}s = Fit runtime")
            self._log(
                log_level,
                f"\t{len(self.features_in)} features in original data used to generate {len(self.features_out)} features in processed data.",
            )

    def print_feature_metadata_info(self, log_level: int = 20):
        """
        Outputs detailed logs of a fit feature selector including the input and output FeatureMetadata objects' feature types.

        Parameters
        ----------
        log_level : int, default 20
            Log level of the logging statements.
        """
        self._log(log_level, "\tTypes of features in original data (raw dtype, special dtypes):")
        self.feature_metadata_in.print_feature_metadata_full(self.log_prefix + "\t\t", log_level=log_level)
        if self.feature_metadata_real:
            self._log(log_level - 5, "\tTypes of features in processed data (exact raw dtype, raw dtype):")
            self.feature_metadata_real.print_feature_metadata_full(
                self.log_prefix + "\t\t", print_only_one_special=True, log_level=log_level - 5
            )
        self._log(log_level, "\tTypes of features in processed data (raw dtype, special dtypes):")
        self.feature_metadata.print_feature_metadata_full(self.log_prefix + "\t\t", log_level=log_level)

    def save(self, path: str):
        save_pkl.save(path=path, object=self)

    def _more_tags(self) -> dict:
        """
        Special values to enable advanced functionality.

        Tags
        ----
        feature_interactions : bool, default True
            If True, then treat all features_out as if they depend on all features_in.
            If False, then treat each features_out as if it was generated by a 1:1 mapping (no feature interactions).
                This enables advanced functionality regarding automated feature pruning, but is only valid for selectors which only transform each feature
                and do not perform interactions.
        allow_post_selectors : bool, default True
            If False, will raise an AssertionError if post_selectors is specified during init.
                This is reserved for very simple selectors where including post_selectors would not be sensible, such as in RenameFeatureGenerator.
        """
        return {}

    def get_tags(self) -> dict:
        """Gets the tags for this selector."""
        collected_tags = {}
        for base_class in reversed(inspect.getmro(self.__class__)):
            if hasattr(base_class, "_more_tags"):
                # need the if because mixins might not have _more_tags
                # but might do redundant work in estimators
                # (i.e. calling more tags on BaseEstimator multiple times)
                more_tags = base_class._more_tags(self)
                collected_tags.update(more_tags)
        return collected_tags

    def _register_fit_metadata(self, **kwargs):
        """
        Used to track properties of the inputs received during fit, such as if validation data was present.
        """
        if not self._is_fit_metadata_registered:
            self._fit_metadata = self._compute_fit_metadata(**kwargs)
            self._is_fit_metadata_registered = True

    def _compute_fit_metadata(self, X: pd.DataFrame = None, X_val: pd.DataFrame = None,
                              X_unlabeled: pd.DataFrame = None, num_cpus: int = None, num_gpus: int = None,
                              **kwargs) -> dict:
        fit_metadata = dict(num_samples=len(X) if X is not None else None, val_in_fit=X_val is not None,
                            unlabeled_in_fit=X_unlabeled is not None, num_cpus=num_cpus, num_gpus=num_gpus)
        return fit_metadata

    def _init_params(self):
        """Initializes model hyperparameters"""
        hyperparameters = self._user_params
        self._set_default_params()
        self.nondefault_params = []
        if hyperparameters is not None:
            self.params.update(hyperparameters)
            self.nondefault_params = list(hyperparameters.keys())[
                                     :]  # These are hyperparameters that user has specified.
        self.params_trained = dict()
        self._validate_params()

    @classmethod
    def _init_user_params(
            cls, params: dict[str, Any] | None = None, ag_args_fit: str = AG_ARGS_FIT,
            ag_arg_prefix: str = AG_ARG_PREFIX
    ) -> (dict[str, Any], dict[str, Any]):
        """
        Given the user-specified hyperparameters, split into `params` and `params_aux`.

        Parameters
        ----------
        params : dict[str, Any], default = None
            The model hyperparameters dictionary
        ag_args_fit : str, default = "ag_args_fit"
            The params key to look for that contains params_aux.
            If the key is present, the value is used for params_aux and popped from params.
            If no such key is found, then initialize params_aux as an empty dictionary.
        ag_arg_prefix : str, default = "ag."
            The key prefix to look for that indicates a parameter is intended for params_aux.
            If None, this logic is skipped.
            If a key starts with this prefix, it is popped from params and added to params_aux with the prefix removed.
            For example:
                input:  params={'ag.foo': 2, 'abc': 7}, params_aux={'bar': 3}, and ag_arg_prefix='.ag',
                output: params={'abc': 7}, params_aux={'bar': 3, 'foo': 2}
            In cases where the key is specified multiple times, the value of the key with the prefix will always take priority.
            A warning will be logged if a key is present multiple times.
            For example, given the most complex scenario:
                input:  params={'ag.foo': 1, 'foo': 2, 'ag_args_fit': {'ag.foo': 3, 'foo': 4}}
                output: params={'foo': 2}, params_aux={'foo': 1}

        Returns
        -------
        params, params_aux : (dict[str, Any], dict[str, Any])
            params will contain the native model hyperparameters
            params_aux will contain special auxiliary hyperparameters
        """
        params = copy.deepcopy(params) if params is not None else dict()
        assert isinstance(params, dict), f"Invalid dtype of params! Expected dict, but got {type(params)}"
        for k in params.keys():
            if not isinstance(k, str):
                logger.warning(
                    f"Warning: Specified {cls.__name__} hyperparameter key is not of type str: {k} (type={type(k)}). "
                    f"There might be a bug in your configuration."
                )

        params_aux = params.pop(ag_args_fit, dict())
        if params_aux is None:
            params_aux = dict()
        assert isinstance(params_aux, dict), f"Invalid dtype of params_aux! Expected dict, but got {type(params_aux)}"
        if ag_arg_prefix is not None:
            param_aux_keys = list(params_aux.keys())
            for k in param_aux_keys:
                if isinstance(k, str) and k.startswith(ag_arg_prefix):
                    k_no_prefix = k[len(ag_arg_prefix):]
                    if k_no_prefix in params_aux:
                        logger.warning(
                            f'Warning: {cls.__name__} hyperparameter "{k}" is present '
                            f'in `ag_args_fit` as both "{k}" and "{k_no_prefix}". '
                            f'Will use "{k}" and ignore "{k_no_prefix}".'
                        )
                    params_aux[k_no_prefix] = params_aux.pop(k)
            param_keys = list(params.keys())
            for k in param_keys:
                if isinstance(k, str) and k.startswith(ag_arg_prefix):
                    k_no_prefix = k[len(ag_arg_prefix):]
                    if k_no_prefix in params_aux:
                        logger.warning(
                            f'Warning: {cls.__name__} hyperparameter "{k}" is present '
                            f"in both `ag_args_fit` and `hyperparameters`. "
                            f"Will use `hyperparameters` value."
                        )
                    params_aux[k_no_prefix] = params.pop(k)
        return params, params_aux

    def _init_params_aux(self):
        """
        Initializes auxiliary hyperparameters.
        These parameters are generally not model specific and can have a wide variety of effects.
        For documentation on some of the available options and their defaults, refer to `self._get_default_auxiliary_params`.
        """
        hyperparameters_aux = self._user_params_aux
        self._set_default_auxiliary_params()
        if hyperparameters_aux is not None:
            self.params_aux.update(hyperparameters_aux)
        self._validate_params_aux()

    def _get_params_aux(self) -> dict:
        hyperparameters_aux = self._user_params_aux
        default_auxiliary_params = self._get_default_auxiliary_params()
        if hyperparameters_aux is not None:
            default_auxiliary_params.update(hyperparameters_aux)
        return default_auxiliary_params

    # TODO: Consider validating before fit call to avoid executing a ray task when it will immediately fail this check in distributed mode
    # TODO: Consider avoiding logging `Fitting model: xyz...` if this fails for particular error types.
    def _validate_params(self):
        """
        Verify correctness of self.params
        """
        pass

    def _validate_params_aux(self):
        """
        Verify correctness of self.params_aux
        """
        if "num_cpus" in self.params_aux:
            num_cpus = self.params_aux["num_cpus"]
            if num_cpus is not None and not isinstance(num_cpus, int):
                raise TypeError(f"`num_cpus` must be an int or None. Found: {type(num_cpus)} | Value: {num_cpus}")

    def can_estimate_memory_usage_static(self) -> bool:
        """
        True if `estimate_memory_usage_static` is implemented for this model.
        If False, calling `estimate_memory_usage_static` will raise a NotImplementedError.
        """
        return self._get_class_tags().get("can_estimate_memory_usage_static", False)

    @classmethod
    def _get_class_tags(cls) -> dict:
        """
        Class tags are tags assigned to a class that are fixed.
        These can be accessed prior to initializing an object.
        Tags are used for identifying if an object supports certain functionality.
        """
        collected_tags = {}
        for base_class in reversed(inspect.getmro(cls)):
            if hasattr(base_class, "_class_tags"):
                # need the if because mixins might not have _class_tags
                # but might do redundant work in estimators
                # (i.e. calling more tags on BaseEstimator multiple times)
                more_tags = base_class._class_tags()
                collected_tags.update(more_tags)
        return collected_tags

    def can_estimate_memory_usage_static_child(self) -> bool:
        """
        True if `estimate_memory_usage_static` is implemented for this model's child.
        If False, calling `estimate_memory_usage_static_child` will raise a NotImplementedError.
        """
        return self.can_estimate_memory_usage_static()

    # TODO: v0.1 update to be aligned with _set_default_auxiliary_params(), add _get_default_params()
    def _set_default_params(self):
        pass

    def _set_default_auxiliary_params(self):
        """
        Sets the default aux parameters of the model.
        This method should not be extended by inheriting models, instead extend _get_default_auxiliary_params.
        """
        # TODO: Consider adding to get_info() output
        default_auxiliary_params = self._get_default_auxiliary_params()
        for key, value in default_auxiliary_params.items():
            self._set_default_param_value(key, value, params=self.params_aux)

    # TODO: v0.1 consider adding documentation to each model highlighting which feature dtypes are valid
    def _get_default_auxiliary_params(self) -> dict:
        """
        Dictionary of auxiliary parameters that dictate various model-agnostic logic, such as:
            Which column dtypes are filtered out of the input data, or how much memory the model is allowed to use.
        """
        default_auxiliary_params = dict(
            max_memory_usage_ratio=1.0,
            # Ratio of memory usage allowed by the model. Values > 1.0 have an increased risk of causing OOM errors. Used in memory checks during model training to avoid OOM errors.
            max_time_limit_ratio=1.0,
            max_time_limit=None,
            min_time_limit=0,
            # min time_limit value during fit(). If the provided time_limit is less than this value, it will be replaced by min_time_limit. Occurs after max_time_limit is applied.
            valid_raw_types=None,  # If a feature's raw type is not in this list, it is pruned.
            valid_special_types=None,  # If a feature has a special type not in this list, it is pruned.
            ignored_type_group_special=None,
            # List, drops any features in `self.feature_metadata.type_group_map_special[type]` for type in `ignored_type_group_special`. | Currently undocumented in task.
            ignored_type_group_raw=None,
            # List, drops any features in `self.feature_metadata.type_group_map_raw[type]` for type in `ignored_type_group_raw`. | Currently undocumented in task.
            # Kwargs for `autogluon.tabular.features.feature_metadata.FeatureMetadata.get_features()`.
            #  Overrides valid_raw_types, valid_special_types, ignored_type_group_special and ignored_type_group_raw. | Currently undocumented in task.
            get_features_kwargs=None,
            get_features_kwargs_extra=None,
            # If not None, applies an additional feature filter to the result of get_feature_kwargs. This should be reserved for users and be None by default. | Currently undocumented in task.
            predict_1_batch_size=None,
            # If not None, calculates `self.predict_1_time` at end of fit call by predicting on this many rows of data.
            temperature_scalar=None,
            # Temperature scaling parameter that is set post-fit if calibrate=True during TabularPredictor.fit() on the model with the best validation score and eval_metric="log_loss".
        )
        return default_auxiliary_params

    def _set_default_param_value(self, param_name, param_value, params=None):
        if params is None:
            params = self.params
        if param_name not in params:
            params[param_name] = param_value

    def _preprocess_fit_args(self, **kwargs) -> dict:
        time_limit = kwargs.get("time_limit", None)
        time_limit_og = time_limit
        max_time_limit_ratio = self.params_aux.get("max_time_limit_ratio", 1)
        if time_limit is not None:
            time_limit *= max_time_limit_ratio
        max_time_limit = self.params_aux.get("max_time_limit", None)
        if max_time_limit is not None:
            if time_limit is None:
                time_limit = max_time_limit
            else:
                time_limit = min(time_limit, max_time_limit)
        min_time_limit = self.params_aux.get("min_time_limit", 0)
        if min_time_limit is None:
            time_limit = min_time_limit
        elif time_limit is not None:
            time_limit = max(time_limit, min_time_limit)
        kwargs["time_limit"] = time_limit
        if time_limit_og != time_limit:
            time_limit_og_str = f"{time_limit_og:.2f}s" if time_limit_og is not None else "None"
            time_limit_str = f"{time_limit:.2f}s" if time_limit is not None else "None"
            logger.log(
                20,
                f"\tTime limit adjusted due to model hyperparameters: "
                f"{time_limit_og_str} -> {time_limit_str} "
                f"(ag.max_time_limit={max_time_limit}, "
                f"ag.max_time_limit_ratio={max_time_limit_ratio}, "
                f"ag.min_time_limit={min_time_limit})",
            )
        kwargs = self._preprocess_fit_resources(**kwargs)
        return kwargs

    def initialize(self, **kwargs) -> dict:
        if not self._is_initialized:
            self._initialize(**kwargs)
            self._is_initialized = True

        kwargs.pop("feature_metadata", None)
        kwargs.pop("num_classes", None)
        kwargs.pop("random_seed", None)
        return kwargs

    def _initialize(self, X=None, y=None, feature_metadata=None, num_classes=None, **kwargs):
        if num_classes is not None:
            self.num_classes = num_classes
        if y is not None:
            if self.problem_type is None:
                self.problem_type = self._infer_problem_type(y=y)
            if self.num_classes is None:
                self.num_classes = self._infer_num_classes(y=y, problem_type=self.problem_type)
        self._init_params_aux()
        self._init_params()
        self.params = self.init_random_seed(random_seed=kwargs.get("random_seed", "auto"), hyperparameters=self.params)

    @classmethod
    def _infer_problem_type(cls, *, y: pd.Series, silent: bool = True) -> str:
        """Infer the problem_type based on y train"""
        return infer_problem_type(y=y, silent=silent)

    @classmethod
    def _infer_num_classes(cls, *, y: pd.Series, problem_type: str = None) -> int | None:
        """Infer num_classes based on y train"""
        if problem_type is None:
            problem_type = cls._infer_problem_type(y=y, silent=True)
        label_cleaner = LabelCleaner.construct(problem_type=problem_type, y=y)
        return label_cleaner.num_classes

    def _process_user_provided_resource_requirement_to_calculate_total_resource_when_ensemble(
            self, system_resource, user_specified_total_resource, user_specified_ensemble_resource, resource_type,
            k_fold
    ):
        if user_specified_total_resource == "auto":
            user_specified_total_resource = math.inf

        # retrieve model level requirement when self is bagged model
        user_specified_model_level_resource = self._get_child_aux_val(key=resource_type, default=None)
        if user_specified_model_level_resource is not None and not isinstance(user_specified_model_level_resource,
                                                                              (int, float)):
            raise TypeError(
                f"{resource_type} must be int or float. Found: {type(user_specified_model_level_resource)} | Value: {user_specified_model_level_resource}"
            )
        if user_specified_model_level_resource is not None:
            assert user_specified_model_level_resource <= system_resource, f"Specified {resource_type} per model base is more than the total: {system_resource}"
        user_specified_lower_level_resource = user_specified_ensemble_resource
        if user_specified_ensemble_resource is not None:
            if user_specified_model_level_resource is not None:
                user_specified_lower_level_resource = min(
                    user_specified_model_level_resource * k_fold, user_specified_ensemble_resource, system_resource,
                    user_specified_total_resource
                )
        else:
            if user_specified_model_level_resource is not None:
                user_specified_lower_level_resource = min(user_specified_model_level_resource * k_fold, system_resource,
                                                          user_specified_total_resource)
        return user_specified_lower_level_resource

    def _calculate_total_resources(
            self, silent: bool = False, total_resources: dict[str, int | float] | None = None,
            parallel_hpo: bool = False, **kwargs
    ) -> dict[str, Any]:
        """
        Process user-specified total resources.
        Sanity checks will be done to user-specified total resources to make sure it's legit.
        When user-specified resources are not defined, will instead look at model's default resource requirements.

        Will set the calculated total resources in kwargs and return it
        """
        resource_manager = get_resource_manager()
        system_num_cpus = resource_manager.get_cpu_count()
        system_num_gpus = resource_manager.get_gpu_count()
        if total_resources is None:
            total_resources = {}
        num_cpus = total_resources.get("num_cpus", "auto")
        num_gpus = total_resources.get("num_gpus", "auto")
        default_num_cpus, default_num_gpus = self._get_default_resources()
        # This could be resource requirement for bagged model or individual model
        user_specified_lower_level_num_cpus = self._user_params_aux.get("num_cpus", None)
        user_specified_lower_level_num_gpus = self._user_params_aux.get("num_gpus", None)
        if user_specified_lower_level_num_cpus is not None:
            assert (
                    user_specified_lower_level_num_cpus <= system_num_cpus
            ), f"Specified num_cpus per {self.__class__.__name__} is more than the total: {system_num_cpus}"
        if user_specified_lower_level_num_gpus is not None:
            assert (
                    user_specified_lower_level_num_gpus <= system_num_gpus
            ), f"Specified num_gpus per {self.__class__.__name__} is more than the total: {system_num_gpus}"
        k_fold = kwargs.get("k_fold", None)
        k_fold = 1 if self.params.get("use_child_oof", False) else k_fold
        if k_fold is not None and k_fold > 0:
            # bagged model will look ag_args_ensemble and ag_args_fit internally to determine resources
            # pass all resources here by default
            default_num_cpus = system_num_cpus
            default_num_gpus = system_num_gpus if default_num_gpus > 0 else 0
            user_specified_lower_level_num_cpus = self._process_user_provided_resource_requirement_to_calculate_total_resource_when_ensemble(
                system_resource=system_num_cpus,
                user_specified_total_resource=num_cpus,
                user_specified_ensemble_resource=user_specified_lower_level_num_cpus,
                resource_type="num_cpus",
                k_fold=k_fold,
            )
            user_specified_lower_level_num_gpus = self._process_user_provided_resource_requirement_to_calculate_total_resource_when_ensemble(
                system_resource=system_num_gpus,
                user_specified_total_resource=num_gpus,
                user_specified_ensemble_resource=user_specified_lower_level_num_gpus,
                resource_type="num_gpus",
                k_fold=k_fold,
            )
        if num_cpus != "auto" and num_cpus > system_num_cpus:
            logger.warning(
                f"Specified total num_cpus: {num_cpus}, but only {system_num_cpus} are available. Will use {system_num_cpus} instead")
            num_cpus = system_num_cpus
        if num_gpus != "auto" and num_gpus > system_num_gpus:
            logger.warning(
                f"Specified total num_gpus: {num_gpus}, but only {system_num_gpus} are available. Will use {system_num_gpus} instead")
            num_gpus = system_num_gpus
        if num_cpus == "auto":
            if user_specified_lower_level_num_cpus is not None:
                if not parallel_hpo:
                    num_cpus = user_specified_lower_level_num_cpus
                else:
                    num_cpus = system_num_cpus
            else:
                if not parallel_hpo:
                    num_cpus = default_num_cpus
                else:
                    num_cpus = system_num_cpus
        else:
            if not parallel_hpo:
                if user_specified_lower_level_num_cpus is not None:
                    assert (
                            user_specified_lower_level_num_cpus <= num_cpus
                    ), f"Specified num_cpus per {self.__class__.__name__} is more than the total specified: {num_cpus}"
                    num_cpus = user_specified_lower_level_num_cpus
        if num_gpus == "auto":
            if user_specified_lower_level_num_gpus is not None:
                if not parallel_hpo:
                    num_gpus = user_specified_lower_level_num_gpus
                else:
                    num_gpus = system_num_gpus if user_specified_lower_level_num_gpus > 0 else 0
            else:
                if not parallel_hpo:
                    num_gpus = default_num_gpus
                else:
                    num_gpus = system_num_gpus if default_num_gpus > 0 else 0
        else:
            if not parallel_hpo:
                if user_specified_lower_level_num_gpus is not None:
                    assert (
                            user_specified_lower_level_num_gpus <= num_gpus
                    ), f"Specified num_gpus per {self.__class__.__name__} is more than the total specified: {num_gpus}"
                    num_gpus = user_specified_lower_level_num_gpus

        minimum_model_resources = self.get_minimum_resources(is_gpu_available=(num_gpus > 0))
        minimum_model_num_cpus = minimum_model_resources.get("num_cpus", 1)
        minimum_model_num_gpus = minimum_model_resources.get("num_gpus", 0)

        maximum_model_resources = self._get_default_resources()
        maximum_model_num_cpus = maximum_model_resources.get("num_cpus", None)
        maximum_model_num_gpus = maximum_model_resources.get("num_gpus", None)

        if maximum_model_num_cpus is not None and maximum_model_num_cpus < num_cpus:
            num_cpus = maximum_model_num_cpus
        if maximum_model_num_gpus is not None and maximum_model_num_gpus < num_gpus:
            num_gpus = maximum_model_num_gpus

        assert system_num_cpus >= num_cpus
        assert system_num_gpus >= num_gpus

        assert (
                system_num_cpus >= minimum_model_num_cpus
        ), f"The total system num_cpus={system_num_cpus} is less than minimum num_cpus={minimum_model_num_cpus} to fit {self.__class__.__name__}. Consider using a machine with more CPUs."
        assert (
                system_num_gpus >= minimum_model_num_gpus
        ), f"The total system num_gpus={system_num_gpus} is less than minimum num_gpus={minimum_model_num_gpus} to fit {self.__class__.__name__}. Consider using a machine with more GPUs."

        assert (
                num_cpus >= minimum_model_num_cpus
        ), f"Specified num_cpus={num_cpus} per {self.__class__.__name__} is less than minimum num_cpus={minimum_model_num_cpus}"
        assert (
                num_gpus >= minimum_model_num_gpus
        ), f"Specified num_gpus={num_gpus} per {self.__class__.__name__} is less than minimum num_gpus={minimum_model_num_gpus}"

        if not isinstance(num_cpus, int):
            raise TypeError(f"`num_cpus` must be an int. Found: {type(num_cpus)} | Value: {num_cpus}")

        kwargs["num_cpus"] = num_cpus
        kwargs["num_gpus"] = num_gpus
        if not silent:
            logger.log(15,
                       f"\tFitting {self.name} with 'num_gpus': {kwargs['num_gpus']}, 'num_cpus': {kwargs['num_cpus']}")

        return kwargs

    def _preprocess_fit_resources(
            self, silent: bool = False, total_resources: dict[str, int | float] | None = None,
            parallel_hpo: bool = False, **kwargs
    ) -> dict[str, Any]:
        """
        This function should be called to process user-specified total resources.
        Sanity checks will be done to user-specified total resources to make sure it's legit.
        When user-specified resources are not defined, will instead look at model's default resource requirements.

        When kwargs contains `num_cpus` and `num_gpus` means this total resources has been calculated by previous layers(i.e. bagged model to model base).
        Will respect this value and check if there's specific maximum resource requirements and enforce those

        Will set the calculated resources in kwargs and return it
        """
        if "num_cpus" in kwargs and "num_gpus" in kwargs:
            # This value will only be passed by autogluon through previous layers(i.e. bagged model to model base).
            # We respect this value with highest priority
            # They should always be set to valid values
            enforced_num_cpus = kwargs.get("num_cpus", None)
            enforced_num_gpus = kwargs.get("num_gpus", None)
            enforced_num_cpus, enforced_num_gpus = self._get_default_resources()
            assert enforced_num_cpus is not None and enforced_num_cpus != "auto" and enforced_num_gpus is not None and enforced_num_gpus != "auto"
            # The logic below is needed because ray cluster is running some process in the backend even when it's ready to be used
            # Trying to use all cores on the machine could lead to resource contention situation
            # TODO: remove this logic if ray team can identify what's going on underneath and how to workaround
            max_resources = self._get_maximum_resources()
            max_num_cpus = max_resources.get("num_cpus", None)
            max_num_gpus = max_resources.get("num_gpus", None)
            if max_num_gpus is not None:
                enforced_num_gpus = min(max_num_gpus, enforced_num_gpus)
            if DistributedContext.is_distributed_mode() and (not DistributedContext.is_shared_network_file_system()):
                minimum_model_resources = self.get_minimum_resources(is_gpu_available=(enforced_num_gpus > 0))
                minimum_model_num_cpus = minimum_model_resources.get("num_cpus", 1)
                enforced_num_cpus = max(minimum_model_num_cpus,
                                        enforced_num_cpus - 2)  # leave some cpu resources for process running by cluster nodes
            if max_num_cpus is not None:
                enforced_num_cpus = min(max_num_cpus, enforced_num_cpus)
            kwargs["num_cpus"] = enforced_num_cpus
            kwargs["num_gpus"] = enforced_num_gpus
            return kwargs

        return self._calculate_total_resources(silent=silent, total_resources=total_resources,
                                               parallel_hpo=parallel_hpo, **kwargs)

    # FIXME: Simply log a message that the model is being skipped instead of logging a traceback.
    def validate_fit_args(self, X: pd.DataFrame, **kwargs):
        """
        Verifies if the fit arguments satisfy the model's constraints.
        Raises an exception if constraints are not satisfied.

        Checks for:
            ag.problem_types
            ag.max_rows
            ag.max_features
            ag.max_classes
            ag.ignore_constraints
        """
        if self.is_initialized():
            ag_params = self._get_ag_params()
        else:
            ag_params = self._get_ag_params(params_aux=self._get_params_aux())

        problem_types: list[str] | None = ag_params.get("problem_types", None)
        max_classes: int | None = ag_params.get("max_classes", None)
        max_rows: int | None = ag_params.get("max_rows", None)
        max_features: int | None = ag_params.get("max_features", None)
        ignore_constraints: bool = ag_params.get("ignore_constraints", False)

        if ignore_constraints:
            # skip all validation checks
            logger.log(15, f"\t`ag.ignore_constraints=True`, skipping sanity checks for model...")
            return

        if problem_types is not None:
            if self.problem_type not in problem_types:
                raise AssertionError(
                    f"ag.problem_types={problem_types} for model '{self.name}', "
                    f"but found '{self.problem_type}' problem_type."
                )
            assert self.problem_type in problem_types
        if max_classes is not None:
            if self.num_classes is not None and self.num_classes > max_classes:
                raise AssertionError(
                    f"ag.max_classes={max_classes} for model '{self.name}', "
                    f"but found {self.num_classes} classes."
                )
        if max_rows is not None:
            n_rows = X.shape[0]
            if n_rows > max_rows:
                raise AssertionError(
                    f"ag.max_rows={max_rows} for model '{self.name}', "
                    f"but found {n_rows} rows."
                )
        if max_features is not None:
            n_max_features = X.shape[1]
            if n_max_features > max_features:
                raise AssertionError(
                    f"ag.max_features={max_features} for model '{self.name}', "
                    f"but found {n_max_features} features."
                )

    # TODO: add model-tag to check if the model can work with `None` random seed?
    # TODO: add check that int seed is smaller than `int(np.iinfo(np.int32).max)`?
    def init_random_seed(self, random_seed: int | None | str, hyperparameters: dict | None = None):
        """Initialize the random seed used by the model by setting `self.random_seed`.

        The random seed can be used to control the randomness of the model (e.g., init, training, etc.).
        By default, AutoGluon's random_seed is 0 to ensure reproducibility. Following convention,
        a random seed can be either an integer or None.

        When using a bagged model, this value differs per fold model. The first fold model uses `model_random_seed`,
        the second uses `model_random_seed + 1`, and the last uses `model_random_seed+n_splits-1` where `n_splits`.
        The start value `model_random_seed` can be set via `ag_args_ensemble` in the model's hyperparameters.

        Parameters
        ----------
        random_seed:
            The random seed passed to `fit`. If "auto", the model will use a default random seed of 0.
            Otherwise, it will set the model's random seed to the provided value.
        hyperparameters
            The hyperparameters of the model, which may or may not contain a random_seed.
            If the hyperparameters contain a random_seed, it will be used to set the model's random seed and
            thus override the random_seed provided in `random_seed`.
        """
        # Set default random seed
        if random_seed == "auto":
            random_seed = self.default_random_seed

        # Overwrite random seed based on hyperparameters, if available
        if hyperparameters is not None:
            hp_rs, seed_name = self._get_random_seed_from_hyperparameters(hyperparameters=hyperparameters)
            if not isinstance(hp_rs, str) and seed_name is not None:
                hyperparameters = hyperparameters.copy()
                random_seed = hyperparameters.pop(seed_name)
                assert random_seed == hp_rs

        if self.seed_name is not None:
            if hyperparameters is None:
                hyperparameters = {}
            else:
                hyperparameters = hyperparameters.copy()
            hyperparameters[self.seed_name] = random_seed
            self.random_seed = hyperparameters[self.seed_name]
        else:
            self.random_seed = random_seed

        return hyperparameters

    def _get_random_seed_from_hyperparameters(self, hyperparameters: dict) -> tuple[int | None | str, str | None]:
        """Extract the random seed from the hyperparameters if available.

        A model implementation may override this method to extract the random seed from the hyperparameters such that
        it is used to init the model's random seed. Otherwise, we default to not being able to extract a random seed
        and use the random seed provided by AutoGluon.

        Parameters
        ----------
        hyperparameters:
            The hyperparameters that may contain a random seed.

        Returns
        -------
        random_seed : int | None | str
            The random seed extracted from the hyperparameters, or "N/A" if not available.
        seed_name: str | None
            The key of the extracted random_seed value, or None if not available.
        """
        if self.seed_name is not None:
            if self.seed_name in hyperparameters:
                return hyperparameters[self.seed_name], self.seed_name
            else:
                for seed_name in self.seed_name_alt:
                    if seed_name in hyperparameters:
                        return hyperparameters[seed_name], seed_name
        return "N/A", None

    @classmethod
    def load(cls, path: str, reset_paths: bool = True, verbose: bool = True):
        """
        Loads the model from disk to memory.

        Parameters
        ----------
        path : str
            Path to the saved model, minus the file name.
            This should generally be a directory path ending with a '/' character (or appropriate path separator value depending on OS).
            The model file is typically located in os.path.join(path, cls.model_file_name).
        reset_paths : bool, default True
            Whether to reset the self.path value of the loaded model to be equal to path.
            It is highly recommended to keep this value as True unless accessing the original self.path value is important.
            If False, the actual valid path and self.path may differ, leading to strange behaviour and potential exceptions if the model needs to load any other files at a later time.
        verbose : bool, default True
            Whether to log the location of the loaded file.

        Returns
        -------
        model : cls
            Loaded model object.
        """
        file_path = os.path.join(path, cls.model_file_name)
        model = load_pkl.load(path=file_path, verbose=verbose)
        if reset_paths:
            model.set_contexts(path)
        if hasattr(model, "_compiler"):
            if model._compiler is not None and not model._compiler.save_in_pkl:
                model.model = model._compiler.load(path=path)
        return model

    def save_learning_curves(self, metrics: str | list[str], curves: dict[dict[str, list[float]]],
                             path: str = None) -> str:
        """
        Saves learning curves to disk.

        Outputted Curve Format:
            out = [
                metrics,
                [
                    [ # log_loss
                        [0.693147, 0.690162, ...], # train
                        [0.693147, 0.690162, ...], # val
                        [0.693147, 0.690162, ...], # test
                    ],
                    [ # accuracy
                        [0.693147, 0.690162, ...], # train
                        [0.693147, 0.690162, ...], # val
                        [0.693147, 0.690162, ...], # test
                    ],
                    [ # f1
                        [0.693147, 0.690162, ...], # train
                        [0.693147, 0.690162, ...], # val
                        [0.693147, 0.690162, ...], # test
                    ],
                ]
            ]

        Parameters
        ----------
        metrics : str or list(str)
            List of all evaluation metrics computed at each iteration of the curve
        curves : dict[dict[str : list[float]]]
            Dictionary of evaluation sets and their learning curve dictionaries.
            Each learning curve dictionary contains evaluation metrics computed at each iteration.
            e.g.
                curves = {
                        "train": {
                            'logloss': [0.693147, 0.690162, ...],
                            'accuracy': [0.500000, 0.400000, ...],
                            'f1': [0.693147, 0.690162, ...]
                        },
                        "val": {...},
                        "test": {...},
                    }

        path : str, default None
            Path where the learning curves are saved, minus the file name.
            This should generally be a directory path ending with a '/' character (or appropriate path separator value depending on OS).
            If None, self.path is used.
            The final curve file is typically saved to os.path.join(path, curves.json).

        Returns
        -------
        path : str
            Path to the saved curves, minus the file name.
        """
        if not self._get_class_tags().get("supports_learning_curves", False):
            raise AssertionError(f"Learning Curves are not supported for model: {self.name}")

        if path is None:
            path = self.path
        if not isinstance(metrics, list):
            metrics = [metrics]
        if len(metrics) == 0:
            raise ValueError("At least one metric must be specified to save generated learning curves.")

        os.makedirs(path, exist_ok=True)
        out = self._make_learning_curves(metrics=metrics, curves=curves)
        file_path = path
        save_json.save(file_path, out)
        return file_path

    def _make_learning_curves(self, metrics: str | list[str], curves: dict[dict[str, list[float]]]) -> list[
        list[str], list[str], list[list[float]]]:
        """
        Parameters
        ----------
        metrics : str or list(str)
            List of all evaluation metrics computed at each iteration of the curve
        curves : dict[dict[str : list[float]]]
            Dictionary of evaluation sets and their learning curve dictionaries.
            Each learning curve dictionary contains evaluation metrics computed at each iteration.
            See Abstract Model's save_learning_curves method for a sample curves input.

        Returns
        -------
        list[list[str], list[str], list[list[float]]]: The generated learning curve artifact.
            if eval set names includes: train, val, or test
            these sets will be placed first in the above order.
        """

        # ensure main eval sets first: train, val, test
        items = []
        order = ["train", "val", "test"]
        for eval_set in order:
            if eval_set in curves:
                items.append((eval_set, curves[eval_set]))
                del curves[eval_set]

        items.extend(curves.items())
        eval_sets, curves = list(zip(*items))

        data = []
        for metric in metrics:
            data.append([c[metric] for c in curves])

        return [eval_sets, metrics, data]

    @classmethod
    def load_learning_curves(cls, path: str) -> list:
        """
        Loads the learning_curve data from disk to memory.

        Parameters
        ----------
        path : str
            Path to the saved model, minus the file name.
            This should generally be a directory path ending with a '/' character (or appropriate path separator value depending on OS).
            The model file is typically located in os.path.join(path, cls.model_file_name).

        Returns
        -------
        learning_curves : list
            Loaded learning curve data.
        """
        if not cls._get_class_tags().get("supports_learning_curves", False):
            raise AssertionError("Attempted to load learning curves from model without learning curve support")

        file = path

        if not os.path.exists(file):
            raise FileNotFoundError(
                f"Could not find learning curve file at {file}" + "\nDid you call predictor.fit() with an appropriate learning_curves parameter?"
            )

        return load_json.load(file)

    # TODO: v1.0: Add docs

    # FIXME: This won't work for all models, and self._features is not
    # a trustworthy variable for final input shape
    def _get_input_types(self, batch_size=None) -> list:
        """
        Get input types as a list of tuples, containing shape and dtype.
        This can be useful for building the input_types argument for
        model compilation. This method can be overloaded in derived classes,
        in order to satisfy class-specific requirements.

        Parameters
        ----------
        batch_size : int, default=None
            The batch size for all returned input types.

        Returns
        -------
        List of (shape: tuple[int], dtype: Any)
        shape: tuple[int]
            A tuple that describes input
        dtype: Any, default=np.float32
            The element type in numpy dtype.
        """
        return [((batch_size, len(self._features)), np.float32)]

    def get_hyperparameters_init(self) -> dict:
        """

        Returns
        -------
        hyperparameters: dict
            The dictionary of user specified hyperparameters for the model.

        """
        hyperparameters = self._user_params.copy()
        if self._user_params_aux:
            hyperparameters[AG_ARGS_FIT] = self._user_params_aux.copy()
        return hyperparameters

    def convert_to_template(self):
        """
        After calling this function, returned model should be able to be fit as if it was new, as well as deep-copied.
        The model name and path will be identical to the original, and must be renamed prior to training to avoid overwriting the original model files if they exist.
        """

        params = self.get_params()
        template = self.__class__(**params)

        return template

    @property
    def _path_v2(self) -> str:
        """Path as a property, replace old path logic with this eventually"""
        return self.path_root + self.path_suffix

    @property
    def path_suffix(self) -> str:
        return self.name

    def get_features(self) -> list[str]:
        assert self.is_fit(), "The model must be fit before calling the get_features method."
        if self.feature_metadata:
            return self.feature_metadata.get_features()
        else:
            return self.features

    def get_params(self) -> dict:
        """Get params of the model at the time of initialization"""
        name = self.name
        path = self.path_root
        problem_type = self.problem_type
        hyperparameters = self.get_hyperparameters_init()

        args = dict(
            path=path,
            name=name,
            problem_type=problem_type,
            hyperparameters=hyperparameters,
        )

        return args

    def get_memory_size(self, allow_exception: bool = False) -> int | None:
        """
        Pickled the model object (self) and returns the size in bytes.
        Will raise an exception if `self` cannot be pickled.

        Note: This will temporarily double the memory usage of the model, as both the original and the pickled version will exist in memory.
        This can lead to an out-of-memory error if the model is larger than the remaining available memory.

        Parameters
        ----------
        allow_exception: bool, default = False
            If True and an exception occurs during the memory size calculation, will return None instead of raising the exception.
            For example, if a model failed during fit and had a messy internal state, and then `get_memory_size` was called,
            it may still contain a non-serializable object. By setting `allow_exception=True`, we avoid crashing in this scenario.
            For example: "AttributeError: Can't pickle local object 'func_generator.<locals>.custom_metric'"

        Returns
        -------
        memory_size: int | None
            The memory size in bytes of the pickled model object.
            None if an exception occurred and `allow_exception=True`.
        """
        if allow_exception:
            try:
                return self._get_memory_size()
            except:
                return None
        else:
            return self._get_memory_size()

    def _get_memory_size(self) -> int:
        gc.collect()  # Try to avoid OOM error
        return sys.getsizeof(pickle.dumps(self, protocol=4))

    def estimate_memory_usage(self, X: pd.DataFrame, **kwargs) -> int:
        """
        Estimates the peak memory usage of the model while training.

        Parameters
        ----------
        X: pd.DataFrame
            The training data features

        Returns
        -------
        int: estimated peak memory usage in bytes during training
        """
        assert self.is_initialized(), "Only estimate memory usage after the model is initialized."
        return self._estimate_memory_usage(X=X, **kwargs)

    @classmethod
    def estimate_memory_usage_static(
            cls,
            *,
            X: pd.DataFrame,
            y: pd.Series = None,
            hyperparameters: dict = None,
            problem_type: str = "infer",
            num_classes: int | None | str = "infer",
            **kwargs,
    ) -> int:
        """
        Estimates the peak memory usage of the model while training, without having to initialize the model.

        Parameters
        ----------
        X: pd.DataFrame
            The training data features
        y: pd.Series, optional
            The training data ground truth. Must be specified if problem_type or num_classes is unspecified.
        hyperparameters: dict, optional
            The model hyperparameters
        problem_type: str, default = "infer"
            The problem_type. If "infer" will infer based on y.
        num_classes
            The num_classes. If "infer" will infer based on y.
        **kwargs
            Other optional key-word fit arguments that could impact memory usage for the model.

        Returns
        -------
        int: estimated peak memory usage in bytes during training
        """
        if problem_type == "infer":
            problem_type = cls._infer_problem_type(y=y)
        if isinstance(num_classes, str) and num_classes == "infer":
            num_classes = cls._infer_num_classes(y=y, problem_type=problem_type)
        if hyperparameters is None:
            hyperparameters = {}
        hyperparameters = cls._get_model_params_static(hyperparameters=hyperparameters,
                                                       convert_search_spaces_to_default=True)
        return cls._estimate_memory_usage_static(
            X=X,
            y=y,
            hyperparameters=hyperparameters,
            problem_type=problem_type,
            num_classes=num_classes,
            **kwargs
        )

    def estimate_memory_usage_child(self, X: pd.DataFrame, **kwargs) -> int:
        """
        Estimates the peak memory usage of the child model while training.

        If the model is not a bagged model (aka has no children), then will return its personal memory usage estimate.

        Parameters
        ----------
        X: pd.DataFrame
            The training data features
        **kwargs

        Returns
        -------
        int: estimated peak memory usage in bytes during training of the child
        """
        return self.estimate_memory_usage(**kwargs)

    def estimate_memory_usage_static_child(
            self,
            *,
            X: pd.DataFrame,
            y: pd.Series = None,
            hyperparameters: dict = None,
            problem_type: str = "infer",
            num_classes: int | None | str = "infer",
            **kwargs,
    ) -> int:
        """
        Estimates the peak memory usage of the child model while training, without having to initialize the model.

        Note that this method itself is not static, because the child model must be present
        as a variable in the model to call its static memory estimate method.

        To obtain the child memory estimate in a fully static manner, instead directly call the child's `estimate_memory_usage_static` method.

        Parameters
        ----------
        X: pd.DataFrame
            The training data features
        y: pd.Series, optional
            The training data ground truth. Must be specified if problem_type or num_classes is unspecified.
        hyperparameters: dict, optional
            The model hyperparameters
        problem_type: str, default = "infer"
            The problem_type. If "infer" will infer based on y.
        num_classes
            The num_classes. If "infer" will infer based on y.
        **kwargs
            Other optional key-word fit arguments that could impact memory usage for the model.

        Returns
        -------
        int: estimated peak memory usage in bytes during training of the child
        """
        return self.estimate_memory_usage_static(X=X, y=y, hyperparameters=hyperparameters, problem_type=problem_type,
                                                 num_classes=num_classes, **kwargs)

    def validate_fit_resources(self, num_cpus="auto", num_gpus="auto", total_resources=None, **kwargs):
        """
        Verifies that the provided num_cpus and num_gpus (or defaults if not provided) are sufficient to train the model.
        Raises an AssertionError if not sufficient.
        """
        resources = self._preprocess_fit_resources(num_cpus=num_cpus, num_gpus=num_gpus,
                                                   total_resources=total_resources, silent=True)
        self._validate_fit_resources(**resources)

    def _validate_fit_resources(self, **resources):
        res_min = self.get_minimum_resources()
        for resource_name in res_min:
            if resource_name not in resources:
                raise AssertionError(
                    f"Model requires {res_min[resource_name]} {resource_name} to fit, but no available amount was defined.")
            elif res_min[resource_name] > resources[resource_name]:
                raise AssertionError(
                    f"Model requires {res_min[resource_name]} {resource_name} to fit, but {resources[resource_name]} are available.")
        total_resources = resources.get("total_resources", None)
        if total_resources is None:
            total_resources = {}
        for resource_name, resource_value in total_resources.items():
            if resources[resource_name] > resource_value:
                raise AssertionError(
                    f"Specified {resources[resource_name]} {resource_name} to fit, but only {resource_value} are available in total.")

    def get_minimum_resources(self, is_gpu_available: bool = False) -> dict[str, int | float]:
        """
        Parameters
        ----------
        is_gpu_available : bool, default = False
            Whether gpu is available in the system.
            Model that can be trained both on cpu and gpu can decide the minimum resources based on this.

        Returns a dictionary of minimum resource requirements to fit the model.
        Subclass should consider overriding this method if it requires more resources to train.
        If a resource is not part of the output dictionary, it is considered unnecessary.
        Valid keys: 'num_cpus', 'num_gpus'.
        """
        return {
            "num_cpus": 1,
        }

    def _estimate_memory_usage(self, X: pd.DataFrame, **kwargs) -> int:
        """
        Estimates the peak memory usage during model fitting.
        This method simply provides a default implementation. Each model should consider implementing custom memory estimation logic.

        Parameters
        ----------
        X : pd.DataFrame,
            The training data intended to fit the model with.
        **kwargs : dict,
            The `.fit` kwargs.
            Can optionally be used by custom implementations to better estimate memory usage.
            To best understand what kwargs are available, enter a debugger and put a breakpoint in this method to manually inspect the keys.

        Returns
        -------
        The estimated peak memory usage in bytes during model fit.
        """
        return 4 * get_approximate_df_mem_usage(X).sum()

    @classmethod
    def _estimate_memory_usage_static(
            cls,
            *,
            X: pd.DataFrame,
            hyperparameters: dict = None,
            num_classes: int = 1,
            **kwargs,
    ) -> int:
        raise NotImplementedError

    @disable_if_lite_mode(ret=(None, None))
    def _validate_fit_memory_usage(
            self,
            mem_error_threshold: float = 0.9,
            mem_warning_threshold: float = 0.75,
            mem_size_threshold: int = None,
            approx_mem_size_req: int = None,
            available_mem: int = None,
            **kwargs,
    ) -> tuple[int | None, int | None]:
        """
        Asserts that enough memory is available to fit the model

        If not enough memory, will raise NotEnoughMemoryError
        Memory thresholds depend on the `params_aux` hyperparameter `max_memory_usage_ratio`, which generally defaults to 1.
        if `max_memory_usage_ratio=None`, all memory checks are skipped.

        Parameters
        ----------
        mem_error_threshold : float, default = 0.9
            A multiplier to max_memory_usage_ratio to get the max_memory_usage_error_ratio
            If expected memory usage is >max_memory_usage_error_ratio, raise NotEnoughMemoryError
        mem_warning_threshold : float, default = 0.75
            A multiplier to max_memory_usage_ratio to get the max_memory_usage_warning_ratio
            If expected memory usage is >max_memory_usage_error_ratio, raise NotEnoughMemoryError
        mem_size_threshold : int, default = None
            If not None, skips checking available memory if the expected model size is less than `mem_size_threshold` bytes.
            This is used to speed-up training by avoiding the check in cases where the machine almost certainly has sufficient memory.
        approx_mem_size_req: int, default = None
            If specified, will use this value as the overall memory usage estimate instead of calculating within the method.
        available_mem: int, default = None
            If specified, will use this value as the available memory instead of calculating within the method.
        **kwargs : dict,
            Fit time kwargs, including X, y, X_val, and y_val.
            Can be used to customize estimation of memory usage.

        Returns
        -------
        approx_mem_size_req: int | None
            The estimated memory requirement of the model, in bytes
            If None, approx_mem_size_req was not calculated.
        available_mem: int | None
            The available memory of the system, in bytes
            If None, available_mem was not calculated.
        """
        max_memory_usage_ratio = self.params_aux["max_memory_usage_ratio"]
        if max_memory_usage_ratio is None:
            return approx_mem_size_req, available_mem  # Skip memory check

        if approx_mem_size_req is None:
            approx_mem_size_req = self.estimate_memory_usage(**kwargs)
        if mem_size_threshold is not None and approx_mem_size_req < (
                mem_size_threshold * min(max_memory_usage_ratio, 1)):
            return approx_mem_size_req, available_mem  # Model is smaller than the min threshold to check available mem

        if available_mem is None:
            available_mem = ResourceManager.get_available_virtual_mem()

        # The expected memory usage percentage of the model during fit
        expected_memory_usage_ratio = approx_mem_size_req / available_mem

        # The minimum `max_memory_usage_ratio` values required to avoid an error/warning
        min_error_memory_ratio = expected_memory_usage_ratio / mem_error_threshold
        min_warning_memory_ratio = expected_memory_usage_ratio / mem_warning_threshold

        # The max allowed `expected_memory_usage_ratio` values to avoid an error/warning
        max_memory_usage_error_ratio = mem_error_threshold * max_memory_usage_ratio
        max_memory_usage_warning_ratio = mem_warning_threshold * max_memory_usage_ratio

        log_ag_args_fit_example = '`predictor.fit(..., ag_args_fit={"ag.max_memory_usage_ratio": VALUE})`'
        log_ag_args_fit_example = f"\n\t\tTo set the same value for all models, do the following when calling predictor.fit: {log_ag_args_fit_example}"

        log_user_guideline = (
            f"Estimated to require {approx_mem_size_req / (1024 ** 3):.3f} GB "
            f"out of {available_mem / (1024 ** 3):.3f} GB available memory ({expected_memory_usage_ratio * 100:.3f}%)... "
            f"({max_memory_usage_error_ratio * 100:.3f}% of avail memory is the max safe size)"
        )
        if expected_memory_usage_ratio > max_memory_usage_error_ratio:
            log_user_guideline += (
                f'\n\tTo force training the model, specify the model hyperparameter "ag.max_memory_usage_ratio" to a larger value '
                f"(currently {max_memory_usage_ratio}, set to >={min_error_memory_ratio + 0.05:.2f} to avoid the error)"
                f"{log_ag_args_fit_example}"
            )
            if min_error_memory_ratio >= 1:
                log_user_guideline += (
                    f'\n\t\tSetting "ag.max_memory_usage_ratio" to values above 1 may result in out-of-memory errors. '
                    f"You may consider using a machine with more memory as a safer alternative."
                )
            logger.warning(f"\tWarning: Not enough memory to safely train model. {log_user_guideline}")
            raise NotEnoughMemoryError
        elif expected_memory_usage_ratio > max_memory_usage_warning_ratio:
            log_user_guideline += (
                f'\n\tTo avoid this warning, specify the model hyperparameter "ag.max_memory_usage_ratio" to a larger value '
                f"(currently {max_memory_usage_ratio}, set to >={min_warning_memory_ratio + 0.05:.2f} to avoid the warning)"
                f"{log_ag_args_fit_example}"
            )
            if min_warning_memory_ratio >= 1:
                log_user_guideline += (
                    f'\n\t\tSetting "ag.max_memory_usage_ratio" to values above 1 may result in out-of-memory errors. '
                    f"You may consider using a machine with more memory as a safer alternative."
                )
            logger.warning(f"\tWarning: Potentially not enough memory to safely train model. {log_user_guideline}")

        return approx_mem_size_req, available_mem

    def reduce_memory_size(self, remove_fit: bool = True, remove_info: bool = False, requires_save: bool = True,
                           **kwargs):
        """
        Removes non-essential objects from the model to reduce memory and disk footprint.
        If `remove_fit=True`, enables the removal of variables which are required for fitting the model. If the model is already fully trained, then it is safe to remove these.
        If `remove_info=True`, enables the removal of variables which are used during model.get_info(). The values will be None when calling model.get_info().
        If `requires_save=True`, enables the removal of variables which are part of the model.pkl object, requiring an overwrite of the model to disk if it was previously persisted.

        It is not necessary for models to implement this.
        """
        pass

    def get_info(self, include_feature_metadata: bool = True) -> dict:
        """
        Returns a dictionary of numerous fields describing the model.
        """
        info = {
            "name": self.name,
            "model_type": type(self).__name__,
            "problem_type": self.problem_type,
            "fit_time": self.fit_time,
            "num_classes": self.num_classes,
            "hyperparameters": self.params,
            "hyperparameters_user": self.get_hyperparameters_init(),
            "hyperparameters_fit": self.params_trained,
            # TODO: Explain in docs that this is for hyperparameters that differ in final model from original hyperparameters, such as epochs (from early stopping)
            "hyperparameters_nondefault": self.nondefault_params,
            "num_features": len(self.features) if self.features else None,
            "features": self.features,
            "feature_metadata": self.feature_metadata,
            # 'disk_usage': self.disk_usage(),
            "memory_size": self.get_memory_size(allow_exception=True),  # Memory usage of model in bytes
            "compile_time": self.compile_time if hasattr(self, "compile_time") else None,
            "is_initialized": self.is_initialized(),
            "is_fit": self.is_fit(),
        }
        if self._is_fit_metadata_registered:
            info.update(self._fit_metadata)
        if not include_feature_metadata:
            info.pop("feature_metadata")
        return info

    def save_info(self) -> dict:
        info = self.get_info()

        save_pkl.save(path=os.path.join(self.path, self.model_info_name), object=info)
        json_path = os.path.join(self.path, self.model_info_json_name)
        save_json.save(path=json_path, obj=info)
        return info

    def _get_maximum_resources(self) -> dict[str, int | float]:
        """
        Get the maximum resources allowed to use for this model.
        This can be useful when model not scale well with resources, i.e. cpu cores.
        Return empty dict if no maximum resources needed

        Return
        ------
        dict[str, int | float]
            key, name of the resource, i.e. `num_cpus`, `num_gpus`
            value, maximum amount of resources
        """
        return {}

    def _get_default_resources(self) -> tuple[int, int]:
        """
        Determines the default resource usage of the model during fit.

        Models may want to override this if they depend heavily on GPUs, as the default sets num_gpus to 0.
        """
        num_cpus = ResourceManager.get_cpu_count()
        num_gpus = 0
        return num_cpus, num_gpus

    @classmethod
    def supported_problem_types(cls) -> list[str] | None:
        """
        Returns the list of supported problem types.
        If None is returned, then the model has not specified the supported problem types, and it is unknown which problem types are valid.
            In this case, all problem types are considered supported and the model will never be filtered out based on problem type.
        """
        return None

    # TODO: v1.0 Move params_aux to params, separate logic as in _get_ag_params, keep `ag.` prefix for ag_args_fit params
    #  This will allow to hyperparameter tune ag_args_fit hyperparameters.
    #  Also delete `self.params_aux` entirely, make it a method instead.
    def _get_params(self) -> dict:
        """Gets all params."""
        return self.params.copy()

    def _get_ag_params(self, params_aux: dict | None = None) -> dict:
        """
        Gets params that are not passed to the inner model, but are used by the wrapper.
        These params should exist in `self.params_aux`.
        """
        if params_aux is None:
            params_aux = self.params_aux
        ag_param_names = self._ag_params()
        ag_param_names_common = self._ag_params_common()
        ag_param_names = ag_param_names.union(ag_param_names_common)
        if ag_param_names:
            return {key: val for key, val in params_aux.items() if key in ag_param_names}
        else:
            return dict()

    def _get_model_params(self, convert_search_spaces_to_default: bool = False) -> dict:
        """
        Gets params that are passed to the inner model.

        Parameters
        ----------
        convert_search_spaces_to_default: bool, default = False
            If True, search spaces are converted to the default value.
            This is useful when having to estimate memory usage estimates prior to doing hyperparameter tuning.

        Returns
        -------
        hyperparameters: dict
            Dictionary of model hyperparameters.
        """
        params = self._get_params()
        return self._get_model_params_static(hyperparameters=params,
                                             convert_search_spaces_to_default=convert_search_spaces_to_default)

    @classmethod
    def _get_model_params_static(cls, hyperparameters: dict, convert_search_spaces_to_default: bool = False) -> dict:
        """
        Gets params that are passed to the inner model.
        This is the static version of `_get_model_params`.
        This method can be called prior to initializing the model.

        Parameters
        ----------
        convert_search_spaces_to_default: bool, default = False
            If True, search spaces are converted to the default value.
            This is useful when having to estimate memory usage estimates prior to doing hyperparameter tuning.

        Returns
        -------
        hyperparameters: dict
            Dictionary of model hyperparameters.
        """
        hyperparameters = hyperparameters.copy()
        if convert_search_spaces_to_default:
            for param, val in hyperparameters.items():
                if isinstance(val, Space):
                    hyperparameters[param] = val.default
        return hyperparameters

    # TODO: Add documentation for valid args for each model. Currently only `early_stop`
    def _ag_params(self) -> set[str]:
        """
        Set of params that are not passed to self.model, but are used by the wrapper.
        For developers, this is purely optional and is just for convenience to logically distinguish between model specific parameters and added AutoGluon functionality.
        The goal is to have common parameter names for useful functionality shared between models,
        even if the functionality is not natively available as a parameter in the model itself or under a different name.

        Below are common patterns / options to make available. Their actual usage and options in a particular model should be documented in the model itself, as it has flexibility to differ.

        Possible params:

        generate_curves : bool
            boolean flag determining if learning curves should be saved to disk for iterative learners.

        curve_metrics : list(...)
            list of metrics to be evaluated at each iteration of the learning curves
            (only used if generate_curves is True)

        use_error_for_curve_metrics : bool
            boolean flag determining if learning curve metrics should be displayed in error format (see Scorer class)

        early_stop : int, str, or tuple
            generic name for early stopping logic. Typically can be an int or a str preset/strategy.
            Also possible to pass tuple of (class, kwargs) to construct a custom early stopping object.
                Refer to `autogluon.core.utils.early_stopping` for examples.

        """
        return set()

    @classmethod
    def _ag_params_common(cls) -> set[str]:
        """
        Set of params that are not passed to self.model, but are used by the wrapper.

        These params are available to all models without requiring special handling in the model.
        They are in addition to the params specified in `_ag_params`

        max_rows: int
            If specified, raises an AssertionError at fit time if len(X) > max_rows
        max_features: int
            If specified, raises an AssertionError at fit time if len(X.columns) > max_rows
        max_classes: int
            If specified, raises an AssertionError at fit time if self.num_classes > max_classes
        problem_types: list[str]
            If specified, raises an AssertionError at fit time if self.problem_type not in problem_types
        ignore_constraints: bool
            If True, ignores the values of `max_rows`, `max_features`, `max_classes` and `problem_types`.

        """
        return {
            "max_rows",
            "max_features",
            "max_classes",
            "problem_types",
            "ignore_constraints",
        }

    @property
    def _features(self) -> list[str]:
        return self._features_internal

    def _get_model_base(self):
        return self

    @property
    def fit_num_cpus(self) -> int:
        """Number of CPUs used when this model was fit"""
        return self.get_fit_metadata()["num_cpus"]

    @property
    def fit_num_gpus(self) -> float:
        """Number of GPUs used when this model was fit"""
        return self.get_fit_metadata()["num_gpus"]

    def get_fit_metadata(self) -> dict:
        """
        Returns dictionary of metadata related to model fit that isn't related to hyperparameters.
        Must be called after model has been fit.
        """
        assert self._is_fit_metadata_registered, "fit_metadata must be registered before calling get_fit_metadata()!"
        fit_metadata = dict()
        fit_metadata.update(self._fit_metadata)
        fit_metadata["predict_1_batch_size"] = self._get_child_aux_val(key="predict_1_batch_size", default=None)
        return fit_metadata

    def _get_child_aux_val(self, key: str, default=None):
        """
        Get aux val of child model (or self if no child)
        This is necessary to get a parameter value that is constant across all children without having to load the children after fitting.
        """
        assert self.is_initialized(), "Model must be initialized before calling self._get_child_aux_val!"
        return self.params_aux.get(key, default)

    def is_initialized(self) -> bool:
        """
        Returns True if the model is initialized.
        This indicates whether the model has inferred various information such as problem_type and num_classes.
        A model is automatically initialized when `.fit` or `.hyperparameter_tune` are called.
        """
        return self._is_initialized

    @property
    def fit_num_cpus_child(self) -> int:
        """Number of CPUs used for fitting one model (i.e. a child model)"""
        return self.fit_num_cpus

    @property
    def fit_num_gpus_child(self) -> float:
        """Number of GPUs used for fitting one model (i.e. a child model)"""
        return self.fit_num_gpus

    @classmethod
    def _class_tags(cls) -> dict:
        return {"supports_learning_curves": False}
