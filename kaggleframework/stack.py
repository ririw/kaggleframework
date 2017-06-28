import lightgbm
import luigi
import pandas
import numpy as np
from plumbum import colors

from kaggleframework import feat_forests, raw_forests, demo_dataset
from kaggleframework.framework import fold_max, AutoExitingGBMLike

from sklearn import linear_model, metrics, ensemble


class Stack(luigi.Task):
    @staticmethod
    def datasets(fold):
        return [
            feat_forests.FeatGBT(fold=fold),
            feat_forests.FeatXTC(fold=fold),
            feat_forests.FeatLogit(fold=fold),
            raw_forests.RawGBT(fold=fold),
            raw_forests.RawLogit(fold=fold),
        ]

    def requires(self):
        for fold in range(fold_max):
            for dataset in self.datasets(fold):
                yield dataset
        yield demo_dataset.DemoTargets()

    def output(self):
        return luigi.LocalTarget('cache/stacked_preds.csv')

    def fold_x(self, name, fold):
        xs = []
        for dataset in self.datasets(fold):
            x = dataset.load(name, as_df=True)
            xs.append(x)
        return pandas.concat(xs, 1)

    def run(self):
        for valid_fold in range(fold_max):
            xs = []
            ys = []
            for fold in range(fold_max):
                if fold == valid_fold:
                    continue
                x = self.fold_x('valid', fold)
                xs.append(x)
                ys.append(demo_dataset.DemoTargets().load('valid', fold))
            x = pandas.concat(xs, 0)
            y = np.concatenate(ys)
            cls = AutoExitingGBMLike(lightgbm.LGBMClassifier(num_leaves=512, n_estimators=512),
                                     additional_fit_args={'verbose': False})
            cls.fit(x, y)
            valid_x = self.fold_x('valid', valid_fold)
            valid_y = demo_dataset.DemoTargets().load('valid', valid_fold)
            pred_y = cls.predict_proba(valid_x)
            print(colors.green | 'Stacked logloss (fold {}): {}'.format(valid_fold, metrics.log_loss(valid_y, pred_y)))

        xs = []
        ys = []
        for fold in range(fold_max):
            x = self.fold_x('valid', fold)
            xs.append(x)
            ys.append(demo_dataset.DemoTargets().load('valid', fold))
        x = pandas.concat(xs, 0)
        y = np.concatenate(ys)
        cls = AutoExitingGBMLike(lightgbm.LGBMClassifier(num_leaves=512, n_estimators=512),
                                 additional_fit_args={'verbose': False})
        cls.fit(x, y)

        xs = []
        for fold in range(fold_max):
            x = self.fold_x('test', fold)
            xs.append(x)
        all_preds = cls.predict_proba(pandas.concat(xs, 0))
        test_size = x.shape[0]
        preds = []
        for fold in range(fold_max):
            preds.append(all_preds[fold * test_size:(fold + 1) * test_size])
        preds = np.asarray(preds).mean(0)

        y = demo_dataset.DemoTargets().load('test', None)
        print(colors.green | 'Overall logloss (kaggle score): {}'.format(metrics.log_loss(y, preds)))
