import os

import luigi
import numpy as np
import pandas
from plumbum import colors

from kaggleframework.demo_dataset import DemoDataset, DemoTargets
from kaggleframework.framework import FoldDependent, cache_base, HyperoptimizableParam

from sklearn import ensemble, metrics, linear_model

from kaggleframework.simple_features import MeanFeature, StdFeature, PCAFeature

class FeatLogit(FoldDependent):
    def _load(self, name, as_df):
        res = np.load(self.output().path)[name]
        if as_df:
            return pandas.DataFrame(res, columns=['FeatLogit_{}'.format(i) for i in range(10)])
        else:
            return res

    @staticmethod
    def datasets():
        return [
            MeanFeature(),
            StdFeature(),
            PCAFeature()
        ]

    def requires(self):
        yield DemoTargets()
        for df in self.datasets():
            yield df

    def output(self):
        return luigi.LocalTarget(self.make_path('data.npz'))

    def load_and_cat(self, name):
        fold = None if name == 'test' else self.fold
        res = pandas.concat([
            df.load(name, fold=fold, as_df=True)
            for df in self.datasets()
        ], 1)
        return res

    def run(self):
        self.output().makedirs()
        x_train = self.load_and_cat('train')
        y_train = DemoTargets().load('train', self.fold)

        cls = linear_model.LogisticRegression()
        cls.fit(x_train, y_train)
        x_valid = self.load_and_cat('valid')
        y_valid = DemoTargets().load('valid', self.fold)
        pred_valid = cls.predict_proba(x_valid)

        print(colors.green | 'FeatLogit logloss: {}'.format(metrics.log_loss(y_valid, pred_valid)))
        x_test = self.load_and_cat('test')
        pred_test = cls.predict_proba(x_test)

        np.savez(self.make_path('data.tmp.npz'), valid=pred_valid, test=pred_test)
        os.rename(self.make_path('data.tmp.npz'), self.output().path)

class FeatXTC(FoldDependent):
    def _load(self, name, as_df):
        res = np.load(self.output().path)[name]
        if as_df:
            return pandas.DataFrame(res, columns=['FeatXTC_{}'.format(i) for i in range(10)])
        else:
            return res

    @staticmethod
    def datasets():
        return [
            MeanFeature(),
            StdFeature(),
            PCAFeature()
        ]

    def requires(self):
        yield DemoTargets()
        for df in self.datasets():
            yield df

    max_leaf = HyperoptimizableParam(default=2, distribution=[2, 4, 8, 16])

    def make_path(self, fname):
        name = self.base_name if self.base_name else self.__class__.__name__
        return os.path.join(cache_base,
                            name,
                            'ml_{}'.format(self.max_leaf),
                            str(self.fold), fname)

    def output(self):
        return luigi.LocalTarget(self.make_path('data.npz'))

    def load_and_cat(self, name):
        fold = None if name == 'test' else self.fold
        res = pandas.concat([
            df.load(name, fold=fold, as_df=True)
            for df in self.datasets()
        ], 1)
        return res

    def run(self):
        self.output().makedirs()
        x_train = self.load_and_cat('train')
        y_train = DemoTargets().load('train', self.fold)

        cls = ensemble.ExtraTreesClassifier(n_estimators=512, min_samples_leaf=self.max_leaf, n_jobs=-1)
        cls.fit(x_train, y_train)
        x_valid = self.load_and_cat('valid')
        y_valid = DemoTargets().load('valid', self.fold)
        pred_valid = cls.predict_proba(x_valid)

        print(colors.green | 'FeatXTC logloss: {}'.format(metrics.log_loss(y_valid, pred_valid)))
        print(pandas.Series(cls.feature_importances_, x_train.columns).sort_values())

        x_test = self.load_and_cat('test')
        pred_test = cls.predict_proba(x_test)

        np.savez(self.make_path('data.tmp.npz'), valid=pred_valid, test=pred_test)
        os.rename(self.make_path('data.tmp.npz'), self.output().path)


class FeatGBT(FoldDependent):
    def _load(self, name, as_df):
        res = np.load(self.output().path)[name]
        if as_df:
            return pandas.DataFrame(res, columns=['FeatGBT_{}'.format(i) for i in range(10)])
        else:
            return res

    @staticmethod
    def datasets():
        return [
            MeanFeature(),
            StdFeature(),
            PCAFeature()
        ]

    def requires(self):
        yield DemoTargets()
        for df in self.datasets():
            yield df

    def output(self):
        return luigi.LocalTarget(self.make_path('data.npz'))

    def load_and_cat(self, name):
        fold = None if name == 'test' else self.fold
        res = pandas.concat([
            df.load(name, fold=fold, as_df=True)
            for df in self.datasets()
        ], 1)
        return res

    n_estimators = HyperoptimizableParam(default=128, distribution=[64, 128, 512, 1024])
    max_depth = HyperoptimizableParam(default=11, distribution=[3, 5, 7, 11, 17])

    def make_path(self, fname):
        name = self.base_name if self.base_name else self.__class__.__name__
        return os.path.join(cache_base,
                            name,
                            'n_est_{}'.format(self.n_estimators),
                            'md_{}'.format(self.max_depth),
                            str(self.fold), fname)

    def run(self):
        self.output().makedirs()
        x_train = self.load_and_cat('train')
        y_train = DemoTargets().load('train', self.fold)

        cls = ensemble.GradientBoostingClassifier(n_estimators=512, max_depth=3)
        cls.fit(x_train, y_train)
        x_valid = self.load_and_cat('valid')
        y_valid = DemoTargets().load('valid', self.fold)
        pred_valid = cls.predict_proba(x_valid)

        print(colors.green | 'FeatGBT logloss: {}'.format(metrics.log_loss(y_valid, pred_valid)))
        print(pandas.Series(cls.feature_importances_, x_train.columns).sort_values())

        x_test = self.load_and_cat('test')
        pred_test = cls.predict_proba(x_test)

        np.savez(self.make_path('data.tmp.npz'), valid=pred_valid, test=pred_test)
        os.rename(self.make_path('data.tmp.npz'), self.output().path)
