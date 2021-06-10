import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline, make_union
from tpot.builtins import StackingEstimator
from tpot.export_utils import set_param_recursive

# NOTE: Make sure that the outcome column is labeled 'target' in the data file
tpot_data = pd.read_csv('PATH/TO/DATA/FILE', sep='COLUMN_SEPARATOR', dtype=np.float64)
features = tpot_data.drop('target', axis=1)
training_features, testing_features, training_target, testing_target = \
            train_test_split(features, tpot_data['target'], random_state=1)

# Average CV score on the training set was: 0.9987886525373154
exported_pipeline = make_pipeline(
    StackingEstimator(estimator=LogisticRegression(C=15.0, dual=False, penalty="l2")),
    SGDClassifier(alpha=0.0, eta0=0.1, fit_intercept=True, l1_ratio=0.25, learning_rate="constant", loss="log", penalty="elasticnet", power_t=10.0)
)
# Fix random state for all the steps in exported pipeline
set_param_recursive(exported_pipeline.steps, 'random_state', 1)

exported_pipeline.fit(training_features, training_target)
results = exported_pipeline.predict(testing_features)
