import importlib

from plumbum import colors
from sklearn import metrics

import plumbum.cli
import numpy as np
from sklearn.model_selection import ParameterSampler, ParameterGrid

from kaggleframework.demo_dataset import DemoTargets
from kaggleframework.framework import HyperoptimizableParam, FoldDependent, fold_max


class Hyperoptimizer(plumbum.cli.Application):
    budget = plumbum.cli.SwitchAttr(['-b', '--budget'], argtype=int, default=10)

    def get_cls(self, task):
        module_name = '.'.join(task.split('.')[:-1])
        cls_name = task.split('.')[-1]
        #print('loading: {:s} from {:s}'.format(cls_name, module_name))
        module = importlib.import_module(module_name)
        cls = getattr(module, cls_name)

        return cls

    def build_grid(self, cls):
        dist = {}
        for var in vars(cls):
            v = getattr(cls, var)
            if isinstance(v, HyperoptimizableParam):
                dist[var] = v.distribution
        return dist

    def main(self, task):
        cls = self.get_cls(task)
        assert issubclass(cls, FoldDependent)
        param_grid = self.build_grid(cls)

        try:
            params = list(ParameterSampler(param_grid, n_iter=self.budget))
        except ValueError:
            params = list(ParameterGrid(param_grid))

        scores = []

        for param_list in params:
            #fold = np.random.choice(fold_max//2)
            fold = 0
            param_list['fold'] = fold
            y = DemoTargets().load('valid', fold)
            inst = cls(**param_list)
            assert isinstance(inst, FoldDependent)
            if not inst.complete():
                inst.run()
            pred = inst.load('valid', 0)
            score = metrics.log_loss(y, pred)
            print(colors.yellow | 'Score: {}, Params: {}'.format(param_list, score))

            del param_list['fold']
            scores.append((score, param_list))
        minscore = min(scores, key=lambda v: v[0])
        print('Finished optimization after {} runs'.format(self.budget))
        print('Min score: {:.4f}'.format(minscore[0]))
        print('Attained at params: {}'.format(minscore[1]))


if __name__ == '__main__':
    Hyperoptimizer.run()