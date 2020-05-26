import pandas as pd
from unittest2 import TestCase  # or `from unittest import ...` if on Python 3.4+
import category_encoders.tests.helpers as th
import numpy as np

import category_encoders as encoders


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

    def test_regression(self):
        enc = encoders.PosteriorImputationEncoder(verbose=1, )
        enc.fit(X, y_reg)
        th.verify_numeric(enc.transform(X_t))
        th.verify_numeric(enc.transform(X_t, y_t_reg))
