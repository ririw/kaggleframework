import importlib
import os

import luigi
import nose.tools
import numpy as np
import pandas
import plumbum.cli
from sklearn import model_selection
from sklearn.base import BaseEstimator

fold_max = 10
cache_base = 'cache'


class FoldIndependent(luigi.Task):
    base_name = None

    def load_all(self, name, as_df=False):
        assert name in {'all_train', 'test'}
        if name == 'all_train':
            return self._load(as_df)
        else:
            return self._load_test(as_df)

    def load(self, name, fold, as_df=False):
        assert self.complete(), repr(self) + ' is not complete'
        assert name in {'train', 'test', 'valid'}

        if name == 'test':
            assert fold is None, 'If using test, fold should be None'
            res = self._load_test(as_df)
        else:
            features = self._load(as_df)
            folds = self.folds()
            nose.tools.assert_is_instance(folds, np.ndarray, 'Error while loading: ' + repr(self))
            folds = (folds + fold) % fold_max
            if name == 'valid':
                selection = folds == 0
            else:
                selection = folds != 0
            res = features[selection]

        if as_df:
            nose.tools.assert_is_instance(res, pandas.DataFrame)
        return res

    def _load(self, as_df):
        raise NotImplementedError

    def folds(self):
        raise NotImplementedError

    def _load_test(self, as_df):
        raise NotImplementedError

    def make_path(self, fname):
        name = self.base_name if self.base_name else self.__class__.__name__
        return os.path.join(cache_base, name, fname)


class FoldDependent(luigi.Task):
    base_name = None

    fold = luigi.IntParameter()

    def _load(self, name, as_df):
        raise NotImplementedError

    def load(self, name, as_df=False):
        assert name in {'valid', 'test'}
        res = self._load(name, as_df)
        if as_df:
            nose.tools.assert_is_instance(res, pandas.DataFrame)
        return res

    def make_path(self, fname):
        name = self.base_name if self.base_name else self.__class__.__name__
        return os.path.join(cache_base, name, str(self.fold), fname)

class HyperoptimizableParam(luigi.Parameter):
    def __init__(self, *args, **kwargs):
        self.distribution = kwargs.pop('distribution')
        super().__init__(*args, **kwargs)


class AutoExitingGBMLike(BaseEstimator):
    def __init__(self, cls, additional_fit_args=None):
        self.additional_fit_args = {} if additional_fit_args is None else additional_fit_args
        self.cls = cls

    def fit(self, X, y):
        X_tr, X_te, y_tr, y_te = model_selection.train_test_split(X, y, test_size=0.05)
        self.cls.fit(X_tr, y_tr, eval_set=[(X_te, y_te)], early_stopping_rounds=20, **self.additional_fit_args)

    def predict_proba(self, X):
        return self.cls.predict_proba(X)

    @property
    def feature_importances_(self):
        return self.cls.feature_importances_

    def __repr__(self):
        return 'AutoExitingGBMLike({:s})'.format(repr(self.cls))
