import pandas as pd
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from unittest2 import TestCase  # or `from unittest import ...` if on Python 3.4+
import category_encoders.tests.helpers as th
import numpy as np
from category_encoders.ordinal import OrdinalEncoder

import category_encoders as encoders
from category_encoders.sampling_bayesian import EncoderWrapper

np_X = th.create_array(n_rows=100)
np_X_t = th.create_array(n_rows=50, extras=True)
np_y_reg = np.random.randn(np_X.shape[0])
np_y = np_y_reg > 0.5
np_y_t_reg = np.random.randn(np_X_t.shape[0])
np_y_t = np_y_t_reg > 0.5
X = th.create_dataset(n_rows=100)
X_t = th.create_dataset(n_rows=50, extras=True)
y_reg = pd.DataFrame(np_y_reg)
y_t_reg = pd.DataFrame(np_y_t_reg)
y = pd.DataFrame(np_y)
y_t = pd.DataFrame(np_y_t)


class TestSamplingBayesianEncoder(TestCase):

    def test_classification(self):
        enc = encoders.PosteriorImputationEncoderBC(verbose=1, )
        enc.fit(X, y)
        th.verify_numeric(enc.transform(X_t))
        th.verify_numeric(enc.transform(X_t, y_t))

    def test_regression_numeric(self):
        enc = encoders.SamplingBayesianEncoder(verbose=1, )
        enc.fit(X, y_reg)
        th.verify_numeric(enc.transform(X_t))
        th.verify_numeric(enc.transform(X_t, y_t_reg))

    def test_regression(self):
        enc = encoders.SamplingBayesianEncoder(verbose=1, n_draws=2,
                                               cols=['unique_str', 'invariant', 'underscore', 'none', 'extra', 321,
                                                          'categorical', 'na_categorical', 'categorical_int'])
        X_le = OrdinalEncoder().fit_transform(X).fillna(0)
        inf_values = np.isinf(X_le).sum(axis=1) == 0
        X_le = X_le[inf_values]
        y_le = y[inf_values]
        enc.fit(X_le, y_le)
        X_new = enc.transform(X_le)
        self.assertEqual(len(X_le.columns) + len(enc.cols), len(X_new.columns))
        df_diff = X_new.categorical - X_new.categorical___1
        self.assertTrue(df_diff.max() - df_diff.min() > 0, "Both encoded columns contain the same values")


    def test_wrapper_classification(self):
        enc = encoders.PosteriorImputationEncoderBC(verbose=1, n_draws=2,
                                                    cols=['unique_str', 'invariant', 'underscore', 'none', 'extra', 321,
                                                          'categorical', 'na_categorical', 'categorical_int'])
        classifier = RandomForestClassifier(n_estimators=10)
        wrapper_model = EncoderWrapper(enc, classifier)
        X_le = OrdinalEncoder().fit_transform(X).fillna(0)
        inf_values = np.isinf(X_le).sum(axis=1) == 0
        X_le = X_le[inf_values]
        y_le = y[inf_values]
        self.assertFalse(np.any(np.isnan(X_le)))
        self.assertFalse(np.any(np.isinf(X_le)))
        wrapper_model.fit(X_le, y_le)
        preds = wrapper_model.predict(X_le)
        self.assertEqual(y_le.shape[0], preds.shape[0])

    def test_wrapper_regression(self):
        enc = encoders.SamplingBayesianEncoder(verbose=1, n_draws=2,
                                               cols=['unique_str', 'invariant', 'underscore', 'none', 'extra', 321,
                                                          'categorical', 'na_categorical', 'categorical_int'])
        classifier = RandomForestRegressor(n_estimators=10)
        wrapper_model = EncoderWrapper(enc, classifier)
        X_le = OrdinalEncoder().fit_transform(X).fillna(0)
        inf_values = np.isinf(X_le).sum(axis=1) == 0
        X_le = X_le[inf_values]
        y_le = y_reg[inf_values]
        self.assertFalse(np.any(np.isnan(X_le)))
        self.assertFalse(np.any(np.isinf(X_le)))
        wrapper_model.fit(X_le, y_le)
        preds = wrapper_model.predict(X_le)
        self.assertEqual(y_le.shape[0], preds.shape[0])

