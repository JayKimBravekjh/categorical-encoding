import pandas as pd
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from unittest2 import TestCase  # or `from unittest import ...` if on Python 3.4+
import category_encoders.tests.helpers as th
import numpy as np
from category_encoders.ordinal import OrdinalEncoder

import category_encoders as encoders
from category_encoders.sampling_bayesian import EncoderWrapper, TaskType

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

    def test_classification_numeric(self):
        enc = encoders.SamplingBayesianEncoder(verbose=1, task=TaskType.BINARY_CLASSIFICATION)
        enc.fit(X, y)
        th.verify_numeric(enc.transform(X_t))
        th.verify_numeric(enc.transform(X_t, y_t))

    def test_regression_numeric(self):
        enc = encoders.SamplingBayesianEncoder(verbose=1, task=TaskType.REGRESSION)
        enc.fit(X, y_reg)
        th.verify_numeric(enc.transform(X_t))
        th.verify_numeric(enc.transform(X_t, y_t_reg))

    def test_regression(self):
        enc = encoders.SamplingBayesianEncoder(verbose=1, n_draws=2,
                                               cols=['unique_str', 'invariant', 'underscore', 'none', 'extra', 321,
                                                     'categorical', 'na_categorical', 'categorical_int'],
                                               task=TaskType.REGRESSION)
        X_le = OrdinalEncoder().fit_transform(X).fillna(0)
        inf_values = np.isinf(X_le).sum(axis=1) == 0
        X_le = X_le[inf_values]
        y_le = y_reg[inf_values]
        enc.fit(X_le, y_le)
        X_new = enc.transform(X_le)
        self.assertEqual(len(X_le.columns) + len(enc.cols), len(X_new.columns))
        df_diff = X_new.categorical_encoded_0 - X_new.categorical_encoded_1
        self.assertTrue(df_diff.max() - df_diff.min() > 0, "Both encoded columns contain the same values")

    def test_regression_mean(self):
        enc = encoders.SamplingBayesianEncoder(verbose=1, n_draws=2,
                                               cols=['unique_str', 'invariant', 'underscore', 'none', 'extra', 321,
                                                     'categorical', 'na_categorical', 'categorical_int'],
                                               mapper='mean', task=TaskType.REGRESSION)
        X_le = OrdinalEncoder().fit_transform(X).fillna(0)
        inf_values = np.isinf(X_le).sum(axis=1) == 0
        X_le = X_le[inf_values]
        y_le = y_reg[inf_values]
        enc.fit(X_le, y_le)
        X_new = enc.transform(X_le)
        self.assertEqual(len(X_le.columns), len(X_new.columns))

    def test_regression_custom_mapper(self):
        enc = encoders.SamplingBayesianEncoder(verbose=1, n_draws=2,
                                               cols=['unique_str', 'invariant', 'underscore', 'none', 'extra', 321,
                                                     'categorical', 'na_categorical', 'categorical_int'],
                                               mapper=lambda x: (x[1],), task=TaskType.REGRESSION)
        X_le = OrdinalEncoder().fit_transform(X).fillna(0)
        inf_values = np.isinf(X_le).sum(axis=1) == 0
        X_le = X_le[inf_values]
        y_le = y_reg[inf_values]
        enc.fit(X_le, y_le)
        X_new = enc.transform(X_le)
        self.assertEqual(len(X_le.columns), len(X_new.columns))

    def test_binary_classification(self):
        enc = encoders.SamplingBayesianEncoder(verbose=1, n_draws=2,
                                               cols=['unique_str', 'invariant', 'underscore', 'none', 'extra', 321,
                                                     'categorical', 'na_categorical', 'categorical_int'],
                                               task=TaskType.BINARY_CLASSIFICATION)
        X_le = OrdinalEncoder().fit_transform(X).fillna(0)
        inf_values = np.isinf(X_le).sum(axis=1) == 0
        X_le = X_le[inf_values]
        y_le = y[inf_values]
        enc.fit(X_le, y_le)
        X_new = enc.transform(X_le)
        self.assertEqual(len(X_le.columns), len(X_new.columns))

    def test_binary_classification_mean(self):
        enc = encoders.SamplingBayesianEncoder(verbose=1, n_draws=2,
                                               cols=['unique_str', 'invariant', 'underscore', 'none', 'extra', 321,
                                                     'categorical', 'na_categorical', 'categorical_int'],
                                               task=TaskType.BINARY_CLASSIFICATION, mapper='mean')
        X_le = OrdinalEncoder().fit_transform(X).fillna(0)
        inf_values = np.isinf(X_le).sum(axis=1) == 0
        X_le = X_le[inf_values]
        y_le = y[inf_values]
        enc.fit(X_le, y_le)
        X_new = enc.transform(X_le)
        self.assertEqual(len(X_le.columns), len(X_new.columns))
        #Now we need to make sure that there are no identical values.
        self.check_contains_no_equal(X_new.categorical_encoded_0)


    def test_binary_classification_mean_identity_same(self):
        enc_mean = encoders.SamplingBayesianEncoder(verbose=1, n_draws=2, random_state=578,
                                               cols=['unique_str', 'invariant', 'underscore', 'none', 'extra', 321,
                                                     'categorical', 'na_categorical', 'categorical_int'],
                                               task=TaskType.BINARY_CLASSIFICATION, mapper='mean')
        enc_identity = encoders.SamplingBayesianEncoder(verbose=1, n_draws=2, random_state=578,
                                               cols=['unique_str', 'invariant', 'underscore', 'none', 'extra', 321,
                                                     'categorical', 'na_categorical', 'categorical_int'],
                                               task=TaskType.BINARY_CLASSIFICATION, mapper='identity')

        X_le = OrdinalEncoder().fit_transform(X).fillna(0)
        inf_values = np.isinf(X_le).sum(axis=1) == 0
        X_le = X_le[inf_values]
        y_le = y[inf_values]
        enc_mean.fit(X_le, y_le)
        X_mean = enc_mean.transform(X_le)
        enc_identity.fit(X_le, y_le)
        X_identity = enc_identity.transform(X_le)
        self.assertTrue(X_mean.equals(X_identity))


    def test_binary_classification_woe(self):
        enc = encoders.SamplingBayesianEncoder(verbose=1, n_draws=2,
                                               cols=['unique_str', 'invariant', 'underscore', 'none', 'extra', 321,
                                                     'categorical', 'na_categorical', 'categorical_int'],
                                               task=TaskType.BINARY_CLASSIFICATION, mapper='weight_of_evidence')
        X_le = OrdinalEncoder().fit_transform(X).fillna(0)
        inf_values = np.isinf(X_le).sum(axis=1) == 0
        X_le = X_le[inf_values]
        y_le = y[inf_values]
        enc.fit(X_le, y_le)
        X_new = enc.transform(X_le)
        self.assertEqual(len(X_le.columns), len(X_new.columns))

    def test_binary_classification_custom(self):
        enc = encoders.SamplingBayesianEncoder(verbose=1, n_draws=2,
                                               cols=['unique_str', 'invariant', 'underscore', 'none', 'extra', 321,
                                                     'categorical', 'na_categorical', 'categorical_int'],
                                               task=TaskType.BINARY_CLASSIFICATION, mapper=lambda x: (x[0], x[0] ** 2))
        X_le = OrdinalEncoder().fit_transform(X).fillna(0)
        inf_values = np.isinf(X_le).sum(axis=1) == 0
        X_le = X_le[inf_values]
        y_le = y[inf_values]
        enc.fit(X_le, y_le)
        X_new = enc.transform(X_le)
        self.assertEqual(len(X_le.columns) + len(enc.cols), len(X_new.columns))

    def test_wrapper_classification(self):
        enc = encoders.SamplingBayesianEncoder(verbose=1, n_draws=2,
                                               cols=['unique_str', 'invariant', 'underscore', 'none', 'extra', 321,
                                                     'categorical', 'na_categorical', 'categorical_int'],
                                               task=TaskType.BINARY_CLASSIFICATION)
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
                                                     'categorical', 'na_categorical', 'categorical_int'],
                                               task=TaskType.REGRESSION)
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

    def check_contains_no_equal(self, column: pd.Series):
        for i in range(len(column)-1):
            for j in range(i+1, len(column)):
                self.assertNotAlmostEqual(column.iloc[i], column.iloc[j])
