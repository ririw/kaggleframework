import os

import lightgbm
import luigi
import numpy as np
import pandas
from plumbum import colors
from sklearn import metrics, linear_model

from kaggleframework.demo_dataset import DemoDataset, DemoTargets
from kaggleframework.framework import FoldDependent, HyperoptimizableParam, cache_base, AutoExitingGBMLike


class RawLogit(FoldDependent):
    def _load(self, name, as_df):
        res = np.load(self.output().path)[name]
        if as_df:
            return pandas.DataFrame(res, columns=['RawLogit_{}'.format(i) for i in range(10)])
        else:
            return res

    def make_path(self, fname):
        name = self.base_name if self.base_name else self.__class__.__name__
        return os.path.join(cache_base, name, str(self.fold), fname)

    def requires(self):
        yield DemoDataset()
        yield DemoTargets()

    def output(self):
        return luigi.LocalTarget(self.make_path('data.npz'))

    def run(self):
        self.output().makedirs()
        x_train = DemoDataset().load('train', self.fold)
        y_train = DemoTargets().load('train', self.fold)

        cls = linear_model.LogisticRegression()
        cls.fit(x_train, y_train)
        x_valid = DemoDataset().load('valid', self.fold)
        y_valid = DemoTargets().load('valid', self.fold)
        pred_valid = cls.predict_proba(x_valid)

        print(colors.green | 'RawLogit logloss: {}'.format(metrics.log_loss(y_valid, pred_valid)))

        x_test = DemoDataset().load('test', None)
        pred_test = cls.predict_proba(x_test)

        np.savez(self.make_path('data.tmp.npz'), valid=pred_valid, test=pred_test)
        os.rename(self.make_path('data.tmp.npz'), self.output().path)


class RawGBT(FoldDependent):
    n_estimators = HyperoptimizableParam(default=512, distribution=[64, 128, 512, 1024])
    n_leaves = HyperoptimizableParam(default=15, distribution=[15, 31, 63, 255, 511])

    def _load(self, name, as_df):
        res = np.load(self.output().path)[name]
        if as_df:
            return pandas.DataFrame(res, columns=['RawGBT_{}'.format(i) for i in range(10)])
        else:
            return res

    def make_path(self, fname):
        name = self.base_name if self.base_name else self.__class__.__name__
        return os.path.join(cache_base, name, 'n_est_{}_nl_{}'.format(self.n_estimators, self.n_leaves), str(self.fold), fname)

    def requires(self):
        yield DemoDataset()
        yield DemoTargets()

    def output(self):
        return luigi.LocalTarget(self.make_path('data.npz'))

    def run(self):
        self.output().makedirs()
        x_train = DemoDataset().load('train', self.fold)
        y_train = DemoTargets().load('train', self.fold)

        cls = AutoExitingGBMLike(lightgbm.LGBMClassifier(n_estimators=self.n_estimators, num_leaves=self.n_leaves),
                                 additional_fit_args={'verbose': False})
        cls.fit(x_train, y_train)
        x_valid = DemoDataset().load('valid', self.fold)
        y_valid = DemoTargets().load('valid', self.fold)
        pred_valid = cls.predict_proba(x_valid)

        print(colors.green | 'RawXTC logloss: {}'.format(metrics.log_loss(y_valid, pred_valid)))

        x_test = DemoDataset().load('test', None)
        pred_test = cls.predict_proba(x_test)

        np.savez(self.make_path('data.tmp.npz'), valid=pred_valid, test=pred_test)
        os.rename(self.make_path('data.tmp.npz'), self.output().path)