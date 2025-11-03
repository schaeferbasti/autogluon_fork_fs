"""
Example script for defining and using FeatureSelectors in AutoGluon Tabular.
FeatureSelectors act to clean and prepare the data to maximize predictive accuracy in downstream models.
FeatureSelectors are stateful data preprocessors which take input data (pandas DataFrame) and output transformed data (pandas DataFrame).
FeatureSelectors are first fit on training data through the .fit_transform() function, and then transform new data through the .transform() function.

This example is intended for advanced users that have a strong understanding of feature selection and data preparation.
Most users can get strong performance without specifying custom feature generators due to the generic and powerful default feature generator used by AutoGluon.
An advanced user may wish to create a custom feature selector to:
    1. Experiment with different preprocessing pipelines to improve model quality.
    2. Have full control over what data is being sent to downstream models.
    3. Migrate existing pipelines into AutoGluon for ease of use and deployment.
    4. Contribute new feature selectors to AutoGluon.
"""
################
# Loading Data #
################

from autogluon.tabular import TabularDataset, TabularPredictor

from examples.tabular.example_custom_feature_generator import auto_ml_pipeline_feature_generator, \
    plus_three_feature_generator
from features.src.autogluon.features.generators.selection import FeatureSelector

train_data = TabularDataset('https://autogluon.s3.amazonaws.com/datasets/AdultIncomeBinaryClassification/train_data.csv')  # can be local CSV file as well, returns Pandas DataFrame
test_data = TabularDataset('https://autogluon.s3.amazonaws.com/datasets/AdultIncomeBinaryClassification/test_data.csv')  # another Pandas DataFrame
label = 'class'  # specifies which column do we want to predict
sample_train_data = train_data.head(100)  # subsample for faster demo

# Separate features and labels
# Make sure to not include your label/target column when sending input to the feature generators, or else the label will be transformed as well.
X = sample_train_data.drop(columns=[label])
y = sample_train_data[label]

X_test = test_data.drop(columns=[label])
y_test = test_data[label]

print(X)

##############################
# Fitting feature generators #
##############################

feature_selector = FeatureSelector()

X_transform = feature_selector.fit_transform(X=X, y=y, verbosity=3)  # verbosity=3 to log more information during fit.
X_test_transform = feature_selector.fit_transform(X_test)
print(X_transform.head(5))

###########################################################
# Specifying custom feature generator to TabularPredictor #
###########################################################

example_models = {'GBM': {}, 'CAT': {}}
example_models_2 = {'RF': {}, 'KNN': {}}

# Because auto_ml_pipeline_feature_generator is already fit, it doesn't need to be fit again in predictor. Instead, train_data is just transformed by auto_ml_pipeline_feature_generator.transform(train_data).
# This allows the feature transformation to be completely independent of the training data, we could have used a completely different data source to fit the generator.
predictor = TabularPredictor(label='class').fit(train_data, hyperparameters=example_models, feature_generator=auto_ml_pipeline_feature_generator)
X_test_transform_2 = predictor.transform_features(X_test)  # This is the same as calling auto_ml_pipeline_feature_generator.transform(X_test)
assert(X_test_transform.equals(X_test_transform_2))
# The feature metadata of the feature generator is also preserved. All downstream models will get this feature metadata information to make decisions on how they use the data.
assert(predictor.feature_metadata.to_dict() == auto_ml_pipeline_feature_generator.feature_metadata.to_dict())
predictor.leaderboard(test_data)

# We can train multiple predictors with the same pre-fit feature generator. This can save a lot of time during experimentation if the fitting of the generator is expensive.
predictor_2 = TabularPredictor(label='class').fit(train_data, hyperparameters=example_models_2, feature_generator=auto_ml_pipeline_feature_generator)
predictor_2.leaderboard(test_data)

# We can even specify our custom generator too (although it needs to do a bit more to actually improve the scores, in most situations just use AutoMLPipelineFeatureGenerator)
predictor_3 = TabularPredictor(label='class').fit(train_data, hyperparameters=example_models, feature_generator=plus_three_feature_generator)
predictor_3.leaderboard(test_data)
