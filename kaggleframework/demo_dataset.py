import os

import logging
import luigi

from kaggleframework.framework import FoldIndependent, fold_max

from sklearn import datasets, model_selection
import numpy as np


class DemoDataset(FoldIndependent):
    base_name = 'dataset'

    def output(self):
        return luigi.LocalTarget(self.make_path('data.npz'))

    def _load_test(self, as_df):
        return np.load(self.output().path)['test']

    def folds(self):
        return np.load(self.make_path('folds.npy'))

    def _load(self, as_df):
        return np.load(self.output().path)['train']

    def run(self):
        self.output().makedirs()

        data = datasets.load_digits()
        X = data['data']
        y = data['target']
        train_x, test_x, train_y, test_y = model_selection.train_test_split(X, y)
        folds = np.random.choice(fold_max, size=train_x.shape[0])

        np.save(self.make_path('folds.npy'), folds)

        np.savez(self.make_path('data.tmp.npz'), train=train_x, test=test_x, train_y=train_y, test_y=test_y)
        os.rename(self.make_path('data.tmp.npz'), self.output().path)


class DemoTargets(FoldIndependent):
    def _load_test(self, as_df):
        logging.error('Loading TEST labels, you better not cheat!')
        return np.load(DemoDataset().output().path)['test_y']

    def folds(self):
        return DemoDataset().folds()

    def _load(self, as_df):
        return np.load(DemoDataset().output().path)['train_y']

    def requires(self):
        yield DemoDataset()

    def complete(self):
        return DemoDataset().complete()