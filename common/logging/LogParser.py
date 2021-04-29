from tqdm.auto import tqdm
import numpy as np
import pandas as pd
from common import gsave, gload, count_parameters, dload, call
from common.logging import load_logs, get_df_from_wandb, logs_to_df
import tensorflow.gfile as gfile
from tensorflow.gfile import Glob


class ModelDDLogParser:
    precompute_dir = '/Users/preetum/tmp/parser'

    def __init__(self, proj, log_dir='gs://preetum/logs', model_size='k',
                 precompute_dir=precompute_dir):
        # eg: proj='cifar10-model-dd-50k-p10-sgd1-decay'
        self.proj = proj
        self.log_dir = log_dir
        self.model_size = model_size
        self.precompute_dir = precompute_dir

    def load(self, min_step = 0, try_cache=True):
        ''' Lead parser from logs, filtering to runs which have a _step >= min_step.
        Returns the parser object.
        If try_cache, then returns the precomputed object if it exists.'''

        if try_cache and self.cache_exists():
            return self.load_self()

        self.df = self.get_history()
        self.filter_runs_minstep(min_step=min_step)
        self.compute_from_df()
        return self

    def load_from_wandb(self, proj_name):
        self.df = get_df_from_wandb(proj_name, finished_only=False, model_size = self.model_size)
        self.compute_from_df()


    def compute_from_df(self):
        self.ks = self.get_ks()

        print("Computing slices...")
        df = self.df
        maxKs = df[['model_size', '_step']].groupby('model_size').max().sort_values('model_size')  # the max _step for each k
        maxStep = np.min(maxKs['_step'])  # min_k( max-step_k )
        steps = np.array(sorted(self.df['_step'].unique()))
        self.steps = steps[steps <= maxStep]
        print('Max step:', maxStep)

        self.Ms = self.get_dynamics_grids(metrics=['Test Error', 'Test Loss', 'Train Error', 'Train Loss'],
                                          steps=self.steps,
                                          model_sizes=self.ks)


    def get_step(self, step):
        df = self.df
        return df[df._step == step]

    # def get_minmaxStep(self):
    #     df = self.df
    #     maxKs = df[['k', '_step']].groupby('k').max().sort_values('k')  # the max _step for each k
    #     return np.min(maxKs['_step'])  # min_k( max-step_k )


    def filter_runs(self, run_filter):
        df = self.df
        ids = df['id'].unique()
        df_filt = pd.DataFrame()
        for id in ids:
            df_r = df[df.id == id]
            df_r = run_filter(df_r)
            df_filt = pd.concat([df_filt, df_r])
        self.df = df_filt

    def filter_runs_minstep(self, min_step = 0):
        def run_filter(df_r): # only keep runs with >= min_step steps
            if df_r.query(f'_step >= {min_step}').empty:
                return pd.DataFrame()
            else:
                return df_r
        self.filter_runs(run_filter=run_filter)

    def get_history(self):
        def isvalid(c, s, h):
            return (h is not None)

        proj_glob = Glob(f'{self.log_dir}/{self.proj}/*')
        assert (len(proj_glob) > 0), "Project dir must be non-empty."
        df = pd.DataFrame()
        for run in tqdm(proj_glob):
            config, summary, history = load_logs(run)
            if isvalid(config, summary, history):
                df_r = logs_to_df(config, summary, history)
                df_r['id'] = run
                df = pd.concat([df, df_r])
        df['model_size'] = df[self.model_size]
        return df


    def get_ks(self):
        return sorted(self.df[self.model_size].unique())

    def get_dynamics_grids(self, model_sizes, steps, metrics=['Test Error', 'Test Loss', 'Train Error', 'Train Loss']):
        '''Returns Metric[model_size, step] for each metric'''
        ks = model_sizes
        df = self.df
        Ms = {m: np.zeros((len(ks), len(steps))) for m in metrics}

        for i, k in enumerate(ks):
            d = df[(df['model_size'] == k) & (df['_step'].isin(steps))].sort_values('_step')
            for metric, M in Ms.items():
                M[i, :] = d[metric]
        
        return Ms


    @staticmethod
    def calc_model_params(model_sizes, model_fn):
        nparams = []
        for k in model_sizes:
            nparams.append(count_parameters(model_fn(k)))
        return np.array(nparams)

    @staticmethod
    def load_parser(proj, precompute_dir):
        ''' Loads a parser object from local dir '''
        return gload(f'{precompute_dir}/{proj}_modelDDparser')

    def save_self(self):
        ''' Saves self into local dir '''
        call(f'mkdir -p {self.precompute_dir}')
        gsave(self, f'{self.precompute_dir}/{self.proj}_modelDDparser')

    def load_self(self):
        ''' Loads from into GCS '''
        return self.load_parser(self.proj, self.precompute_dir)

    def cache_exists(self):
        import os
        path = f'{self.precompute_dir}/{self.proj}_modelDDparser'
        return os.path.exists(path)
