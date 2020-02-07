import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

##### Callbacks for change learning rate #####
class ReduceLearningRateCallback(object):
    def __init__(self, monitor_metric, reduce_every=100, ratio=0.5):
        self._monitor_metric = monitor_metric
        self._reduce_every = reduce_every
        self._ratio = ratio
        self._best_score = None
        self._counter = 0

    def _is_higher_score(self, metric_score, is_higher_better):
        if self._best_score is None:
            return True

        if is_higher_better:
            return self._best_score < metric_score
        else:
            return self._best_score > metric_score

    def __call__(self, env):
        evals = env.evaluation_result_list
        lr = env.params['learning_rate']
        for tra_val, name, score, is_higher_better in evals:
            if (env.iteration<100):
                return 
            
            if (name!=self._monitor_metric) | (tra_val!='valid'):
                continue
                
            if not self._is_higher_score(score, is_higher_better):
                self.counter += 1
#                 print(f'\r{self.counter}', end='', flush=True)
                if self.counter==self._reduce_every:
                    new_parameters = {'learning_rate':lr*self._ratio}
                    env.model.reset_parameter(new_parameters)
                    env.params.update(new_parameters)
                    print(f'[{env.iteration}]\tReduce learning rate {lr} -> {env.params["learning_rate"]}')
                    self.counter = 0
                    return
                else:
                    return
            else:
                self._best_score = score
#                 print(f'\r{self._best_score}', end='', flush=True)
                self.counter = 0
                return
            
        raise ValueError('monitoring metric not found')
