from sklearn.base import BaseEstimator, ClassifierMixin, RegressorMixin


class EncoderWrapper(BaseEstimator, ClassifierMixin, RegressorMixin):
    def __init__(self, encoder, estimator):
        self.encoder = encoder
        self.estimator = estimator

    def fit(self, X, y, **kwargs):
        self.encoder.fit(X, y)
        X_transformed = self.encoder.transform(X)
        y_transformed = self.encoder.expand_y(y)
        self.estimator.fit(X_transformed, y_transformed)

    def predict_proba(self, X):
        assert hasattr(self.estimator, 'predict_proba'), '''
            predict_proba() method is not available. You may be dealing with a Regression case 
        '''
        X_transformed = self.encoder.transform(X)
        preds = self.estimator.predict_proba(X_transformed)[:, 1]
        return self.encoder.average_y(preds)

    def predict(self, X):
        if hasattr(self.estimator, 'predict_proba'):
            return self.predict_proba(X).round()
        else:
            X_transformed = self.encoder.transform(X)
            preds = self.estimator.predict(X_transformed)
            return self.encoder.average_y(preds)
