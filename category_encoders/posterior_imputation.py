"""M-probability estimate"""
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from category_encoders.ordinal import OrdinalEncoder
import category_encoders.utils as util
from sklearn.utils.random import check_random_state

__author__ = 'Michael Larionov'


class PosteriorImputationEncoder(BaseEstimator, TransformerMixin):
    """M-probability estimate of likelihood.

    This is a simplified version of target encoder, which goes under names like m-probability estimate or
    additive smoothing with known incidence rates. In comparison to target encoder, m-probability estimate
    has only one tunable parameter (`m`), while target encoder has two tunable parameters (`min_samples_leaf`
    and `smoothing`).

    Parameters
    ----------

    verbose: int
        integer indicating verbosity of the output. 0 for none.
    cols: list
        a list of columns to encode, if None, all string columns will be encoded.
    drop_invariant: bool
        boolean for whether or not to drop encoded columns with 0 variance.
    return_df: bool
        boolean for whether to return a pandas DataFrame from transform (otherwise it will be a numpy array).
    handle_missing: str
        options are 'return_nan', 'error' and 'value', defaults to 'value', which returns the prior probability.
    handle_unknown: str
        options are 'return_nan', 'error' and 'value', defaults to 'value', which returns the prior probability.
    randomized: bool,
        adds normal (Gaussian) distribution noise into training data in order to decrease overfitting (testing data are untouched).
    sigma: float
        standard deviation (spread or "width") of the normal distribution.
    m: float
        this is the "m" in the m-probability estimate. Higher value of m results into stronger shrinking.
        M is non-negative.

    Example
    -------
    >>> from category_encoders import *
    >>> import pandas as pd
    >>> from sklearn.datasets import load_boston
    >>> bunch = load_boston()
    >>> y = bunch.target > 22.5
    >>> X = pd.DataFrame(bunch.data, columns=bunch.feature_names)
    >>> enc = MEstimateEncoder(cols=['CHAS', 'RAD']).fit(X, y)
    >>> numeric_dataset = enc.transform(X)
    >>> print(numeric_dataset.info())
    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 506 entries, 0 to 505
    Data columns (total 13 columns):
    CRIM       506 non-null float64
    ZN         506 non-null float64
    INDUS      506 non-null float64
    CHAS       506 non-null float64
    NOX        506 non-null float64
    RM         506 non-null float64
    AGE        506 non-null float64
    DIS        506 non-null float64
    RAD        506 non-null float64
    TAX        506 non-null float64
    PTRATIO    506 non-null float64
    B          506 non-null float64
    LSTAT      506 non-null float64
    dtypes: float64(13)
    memory usage: 51.5 KB
    None

    References
    ----------

    .. [1] A Preprocessing Scheme for High-Cardinality Categorical Attributes in Classification and Prediction Problems, equation 7, from
    https://dl.acm.org/citation.cfm?id=507538

    .. [2] On estimating probabilities in tree pruning, equation 1, from
    https://link.springer.com/chapter/10.1007/BFb0017010

    .. [3] Additive smoothing, from
    https://en.wikipedia.org/wiki/Additive_smoothing#Generalized_to_the_case_of_known_incidence_rates

    """

    def __init__(self, verbose=0, cols=None, drop_invariant=False, return_df=True,
                 handle_unknown='value', handle_missing='value', random_state=None, 
                 prior_samples_ratio=0.1, n_draws=10, include_precision=True):
        self.verbose = verbose
        self.return_df = return_df
        self.drop_invariant = drop_invariant
        self.drop_cols = []
        self.cols = cols
        self.ordinal_encoder = None
        self._dim = None
        self.mapping = None
        self.handle_unknown = handle_unknown
        self.handle_missing = handle_missing
        self.random_state = random_state
        self.prior_samples_ratio = prior_samples_ratio
        self.feature_names = None
        self.n_draws = n_draws
        self.include_precision = include_precision

    # noinspection PyUnusedLocal
    def fit(self, X, y, **kwargs):
        """Fit encoder according to X and binary y.

        Parameters
        ----------

        X : array-like, shape = [n_samples, n_features]
            Training vectors, where n_samples is the number of samples
            and n_features is the number of features.
        y : array-like, shape = [n_samples]
            Binary target values.

        Returns
        -------

        self : encoder
            Returns self.

        """

        # Unite parameters into pandas types
        X = util.convert_input(X)
        y = util.convert_input_vector(y, X.index).astype(float)

        # The lengths must be equal
        if X.shape[0] != y.shape[0]:
            raise ValueError("The length of X is " + str(X.shape[0]) + " but length of y is " + str(y.shape[0]) + ".")

        self._dim = X.shape[1]

        # If columns aren't passed, just use every string column
        if self.cols is None:
            self.cols = util.get_obj_cols(X)
        else:
            self.cols = util.convert_cols_to_list(self.cols)

        if self.handle_missing == 'error':
            if X[self.cols].isnull().any().any():
                raise ValueError('Columns to be encoded can not contain null')

        self.ordinal_encoder = OrdinalEncoder(
            verbose=self.verbose,
            cols=self.cols,
            handle_unknown='value',
            handle_missing='value'
        )
        self.ordinal_encoder = self.ordinal_encoder.fit(X)
        X_ordinal = self.ordinal_encoder.transform(X)

        # Training
        self.mapping = self._train(X_ordinal, y)

        X_temp = self.transform(X, override_return_df=True)
        self.feature_names = X_temp.columns.tolist()

        # Store column names with approximately constant variance on the training data
        if self.drop_invariant:
            self.drop_cols = []
            generated_cols = util.get_generated_cols(X, X_temp, self.cols)
            self.drop_cols = [x for x in generated_cols if X_temp[x].var() <= 10e-5]
            try:
                [self.feature_names.remove(x) for x in self.drop_cols]
            except KeyError as e:
                if self.verbose > 0:
                    print("Could not remove column from feature names."
                    "Not found in generated cols.\n{}".format(e))
        return self

    def transform(self, X, y=None, override_return_df=False):
        """Perform the transformation to new categorical data.

        When the data are used for model training, it is important to also pass the target in order to apply leave one out.

        Parameters
        ----------

        X : array-like, shape = [n_samples, n_features]
        y : array-like, shape = [n_samples] when transform by leave one out
            None, when transform without target information (such as transform test set)


        Returns
        -------

        p : array, shape = [n_samples, n_numeric + N]
            Transformed values with encoding applied.

        """

        if self.handle_missing == 'error':
            if X[self.cols].isnull().any().any():
                raise ValueError('Columns to be encoded can not contain null')

        if self._dim is None:
            raise ValueError('Must train encoder before it can be used to transform data.')

        # Unite the input into pandas types
        X = util.convert_input(X)

        # Then make sure that it is the right size
        if X.shape[1] != self._dim:
            raise ValueError('Unexpected input dimension %d, expected %d' % (X.shape[1], self._dim,))

        # If we are encoding the training data, we have to check the target
        if y is not None:
            y = util.convert_input_vector(y, X.index).astype(float)
            if X.shape[0] != y.shape[0]:
                raise ValueError("The length of X is " + str(X.shape[0]) + " but length of y is " + str(y.shape[0]) + ".")

        if not list(self.cols):
            return X

        # Do not modify the input argument
        X = X.copy(deep=True)

        X = self.ordinal_encoder.transform(X)

        if self.handle_unknown == 'error':
            if X[self.cols].isin([-1]).any().any():
                raise ValueError('Unexpected categories found in dataframe')

        # Loop over the columns and replace the nominal values with the numbers
        X = self._score(X)

        # Postprocessing
        # Note: We should not even convert these columns.
        if self.drop_invariant:
            for col in self.drop_cols:
                X.drop(col, 1, inplace=True)

        if self.return_df or override_return_df:
            return X
        else:
            return X.values

    def fit_transform(self, X, y=None, **fit_params):
        """
        Encoders that utilize the target must make sure that the training data are transformed with:
            transform(X, y)
        and not with:
            transform(X)
        """

        # the interface requires 'y=None' in the signature but we need 'y'
        if y is None:
            raise(TypeError, 'fit_transform() missing argument: ''y''')

        return self.fit(X, y, **fit_params).transform(X, y)

    
    def _compute_posterior_parameters(self, y_bar, y_var, n, mu_0=0, nu=0, alpha=0, beta=0):
        ss = y_var * (n-1)
        new_mu = (nu * mu_0 + n * y_bar) / (nu + n)
        new_nu = nu + n
        new_alpha = alpha  + n/2
        new_beta = beta + 1/2*ss + n*nu/(n+nu)*(y_bar-mu_0)**2/2
        return new_mu, new_nu, new_alpha, new_beta

    
    def _train(self, X, y):
        # Initialize the output
        mapping = {}

        # Calculate global statistics
        prior = self._compute_posterior_parameters(y.mean(), y.var(), y.shape[0])

        for switch in self.ordinal_encoder.category_mapping:
            col = switch.get('col')
            values = switch.get('mapping')
            # Calculate sum and count of the target for each unique value in the feature col
            stats = y.groupby(X[col]).agg(['mean', 'count', 'var'])  # Count of x_{i,+} and x_i

            # Calculate the m-probability estimate
            estimate = self._compute_posterior_parameters(stats['mean'], stats['var'], stats['count'], 
                prior[0], prior[1]*self.prior_samples_ratio, prior[2]*self.prior_samples_ratio, 
                prior[3]*self.prior_samples_ratio)

            # Ignore unique columns. This helps to prevent overfitting on id-like columns
            singles = estimate[-1].isnull()
            estimate[0][singles] = prior[0]
            estimate[1][singles] = prior[1]
            estimate[2][singles] = prior[2]
            estimate[3][singles] = prior[3]


            if self.handle_unknown == 'return_nan':
                estimate[0].loc[-1] = np.nan
                estimate[1].loc[-1] = np.nan
                estimate[2].loc[-1] = np.nan
                estimate[3].loc[-1] = np.nan
            elif self.handle_unknown == 'value':
                estimate[0].loc[-1] = prior[0]
                estimate[1].loc[-1] = prior[1]
                estimate[2].loc[-1] = prior[2]
                estimate[3].loc[-1] = prior[3]

            if self.handle_missing == 'return_nan':
                estimate[0].loc[values.loc[np.nan]] = np.nan
                estimate[1].loc[values.loc[np.nan]] = np.nan
                estimate[2].loc[values.loc[np.nan]] = np.nan
                estimate[3].loc[values.loc[np.nan]] = np.nan
            elif self.handle_missing == 'value':
                estimate[0].loc[-2] = prior[0]
                estimate[1].loc[-2] = prior[1]
                estimate[2].loc[-2] = prior[2]
                estimate[3].loc[-2] = prior[3]

            # Store the m-probability estimate for transform() function
            mapping[col] = estimate

        return mapping

    def _sample_single(self, mu, lambda_, alpha, beta):
        shape = alpha
        scale = 1/beta
        tau = np.random.gamma(shape, scale)
        x = np.random.normal(mu, 1/np.sqrt(lambda_ * tau))
        return (x, tau)


    def _score_one_draw(self, X_in):
        X = X_in.copy(deep=True)
        sample_function = np.vectorize(self._sample_single)#, signature='(a1),(a2),(a3),(a4)->(k)')
        for col in self.cols:
            # Score the column
            mu = self.mapping[col][0]
            lambda_ = self.mapping[col][1]
            alpha = self.mapping[col][2]
            beta = self.mapping[col][3]
            sample_result_x, sample_result_tau = sample_function(mu, lambda_, alpha, beta)
            sample_result = pd.DataFrame(data=np.vstack([sample_result_x, sample_result_tau]).T, columns=['x', 'tau'], index=mu.index)
            if self.include_precision:
                X[f'precision_{col}'] = X[col].map(sample_result.tau)
            X[col] = X[col].map(sample_result.x)
        return X

    def _score(self, X):
        return pd.concat([self._score_one_draw(X) for _ in range(self.n_draws)])


    def get_feature_names(self):
        """
        Returns the names of all transformed / added columns.

        Returns
        -------
        feature_names: list
            A list with all feature names transformed or added.
            Note: potentially dropped features are not included!

        """
        if not isinstance(self.feature_names, list):
            raise ValueError("Estimator has to be fitted to return feature names.")
        else:
            return self.feature_names

    def expand_y(self, y):
        y = np.array(y)
        return np.vstack([y for _ in range(self.n_draws)])

    def average_y(self, y):
        split_y =  np.split(y, self.n_draws)
        split_y_combined = np.vstack(split_y)
        return split_y_combined.mean(axis=0)
