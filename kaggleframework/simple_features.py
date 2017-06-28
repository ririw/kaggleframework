import os

import luigi
import numpy as np
import pandas
from sklearn import decomposition

from kaggleframework.framework import FoldIndependent
from kaggleframework.demo_dataset import DemoDataset


class MeanFeature(FoldIndependent):
    def _load_test(self, as_df):
        res = np.load(self.output().path)['test']
        if as_df:
            res = pandas.DataFrame({
                'mean': res
            })
        return res

    def folds(self):
        return DemoDataset().folds()

    def _load(self, as_df):
        res = np.load(self.output().path)['train']
        if as_df:
            res = pandas.DataFrame({
                'mean': res
            })
        return res

    def requires(self):
        yield DemoDataset()

    def output(self):
        return luigi.LocalTarget(self.make_path('data.npz'))

    def run(self):
        self.output().makedirs()
        dataset = DemoDataset()
        train = dataset.load_all('all_train')
        test = dataset.load_all('test')
        train_mean = train.mean(1)
        test_mean = test.mean(1)

        np.savez(self.make_path('data.tmp.npz'), train=train_mean, test=test_mean)
        os.rename(self.make_path('data.tmp.npz'), self.output().path)


class StdFeature(FoldIndependent):
    def _load_test(self, as_df):
        data = np.load(self.output().path)['test']
        if as_df:
            data = pandas.Series(data, name='std').to_frame()
        return data

    def _load(self, as_df):
        data = np.load(self.output().path)['train']
        if as_df:
            data = pandas.Series(data, name='std').to_frame()
        return data

    def folds(self):
        return DemoDataset().folds()

    def requires(self):
        yield DemoDataset()

    def output(self):
        return luigi.LocalTarget(self.make_path('data.npz'))

    def run(self):
        self.output().makedirs()
        train = DemoDataset().load_all('all_train')
        test = DemoDataset().load_all('test')

        train_std = train.std(1)
        test_std = test.std(1)

        np.savez(self.make_path('data.tmp.npz'), train=train_std, test=test_std)
        os.rename(self.make_path('data.tmp.npz'), self.output().path)


class PCAFeature(FoldIndependent):
    n_components = 4
    def _load_test(self, as_df):
        data = np.load(self.output().path)['test']
        if as_df:
            data = pandas.DataFrame(data, columns=['pca_{}'.format(i) for i in range(self.n_components)])
        return data

    def _load(self, as_df):
        data = np.load(self.output().path)['train']
        if as_df:
            data = pandas.DataFrame(data, columns=['pca_{}'.format(i) for i in range(self.n_components)])
        return data

    def folds(self):
        return DemoDataset().folds()

    def requires(self):
        yield DemoDataset()

    def output(self):
        return luigi.LocalTarget(self.make_path('data.npz'))

    def run(self):
        self.output().makedirs()
        train = DemoDataset().load_all('all_train')
        test = DemoDataset().load_all('test')
        all_data = np.concatenate([train, test], 0)

        pca = decomposition.PCA(n_components=self.n_components)
        transformed = pca.fit_transform(all_data)
        train_pca = transformed[:train.shape[0]]
        test_pca = transformed[train.shape[0]:]
        assert train_pca.shape[0] == train.shape[0]
        assert test_pca.shape[0] == test.shape[0]
        assert train_pca.shape[1] == self.n_components
        assert test_pca.shape[1] == self.n_components

        np.savez(self.make_path('data.tmp.npz'), train=train_pca, test=test_pca)
        os.rename(self.make_path('data.tmp.npz'), self.output().path)
