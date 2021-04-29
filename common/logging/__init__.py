import os
import subprocess
import pickle
from common import gsave, gload
import tensorflow as tf
import pandas as pd
from tqdm.auto import tqdm

def do_cmd(cmd: str):
    proc = subprocess.Popen(cmd, shell=True)
    proc.communicate()

class Logger():
    def __init__(self, proj_name, wandb, gcs_root="gs://preetum/logs"):
        ''' Logs run config,summmary, and history into GCS.
            For history: Appends logs to a local pickle file, and syncs this to GCS whenever sync() is called'''

        self.step = 0
        self.wandb = wandb
        self.gcs_logdir =   f'{gcs_root}/{proj_name}/{wandb.run.id}'
        print("GCS Logdir:", self.gcs_logdir)
        self.history = []
        self.gcs_history =   f'{self.gcs_logdir}/history'

        self._log_config()

    def _log_config(self):
        path = f'{self.gcs_logdir}/config'
        gsave(dict(self.wandb.config), path)

    def log(self, log_dict):
        log_dict['_step'] = self.step
        self.history.append(log_dict)
        self.step += 1

    def sync(self):
        gsave(self.history, self.gcs_history)
    
    def log_summary(self, log_dict):
        path = f'{self.gcs_logdir}/summary'
        gsave(log_dict, path)

    def save_obj(self, obj, fname):
        ''' Saves obj --> log_dir/fname '''
        path = f'{self.gcs_logdir}/{fname}'
        gsave(obj, path)


def load_logs(gpath):
    ''' Loads config, summary, history as saved by Logger from GCS.
        history is a list of dicts.
        gpath is eg "gs://preetum-rcall/logs/cifar-test-2/149kdoca" '''
    def _gload(path):
        if tf.io.gfile.exists(path):
            return gload(path)
        else:
            return None
    config =  _gload(f'{gpath}/config')
    history = _gload(f'{gpath}/history')
    summary = _gload(f'{gpath}/summary')
    return config, summary, history

def logs_to_df(config, summary, history):
    df = pd.DataFrame.from_records(history)
    for k, v in config.items():
        if type(v) in [str, float, int]:
            df[k] = v
        else:
            df[k] = str(v)  # lists, etc give pandas issues
    return df

def summary_logs_to_df(config, summary):
    df = pd.DataFrame.from_records(summary)
    for k, v in config.items():
        if type(v) in [str, float, int]:
            df[k] = v
        else:
            df[k] = str(v)  # lists, etc give pandas issues
    return df

def get_history(gcs_dir):
    proj_glob = tf.io.gfile.glob(f'{gcs_dir}/*')
    assert (len(proj_glob) > 0), "Project dir must be non-empty."
    df = pd.DataFrame()
    for run in tqdm(proj_glob):
        config, summary, history = load_logs(run)
        df_r = logs_to_df(config, summary, history)
        df = pd.concat([df, df_r])
    return df

def get_summaries(gcs_dir):
    proj_glob = tf.io.gfile.glob(f'{gcs_dir}/*')
    assert (len(proj_glob) > 0), "Project dir must be non-empty."
    df = pd.DataFrame()
    for run in tqdm(proj_glob):
        config, summary, history = load_logs(run)
        df_r = summary_logs_to_df(config, summary)
        df = pd.concat([df, df_r])
    return df

def get_df_from_wandb(project: str, finished_only=False):
    # project: 'preetum/cifar-model-dd-50k-p10'
    import wandb
    api = wandb.Api()
    runs = api.runs(project, per_page=200)
    df = pd.DataFrame()
    for r in tqdm(runs):
        if r.state == 'finished' or not finished_only:
            df_hist = r.history(samples=500000)
            df_hist['name'] = r.name
            df_hist['id'] = r.id
            for k, v in dict(r.config).items():
                if type(v) in [str, float, int, bool]:
                    df_hist[k] = v
                else:
                    df_hist[k] = str(v)

            df = pd.concat([df, df_hist])
    return df


def get_summary_from_wandb(project: str, finished_only=False):
    # project: 'preetum/cifar-model-dd-50k-p10'
    import wandb
    api = wandb.Api()
    runs = api.runs(project, per_page=200)
    dicts = []
    for r in tqdm(runs):
        if r.state == 'finished' or not finished_only:
            rdict = dict(r.summary)
            rdict['name'] = r.name
            rdict['id'] = r.id
            for k, v in dict(r.config).items():
                if type(v) in [str, float, int, bool]:
                    rdict[k] = v
                else:
                    rdict[k] = str(v)

            dicts.append(rdict)

    df = pd.DataFrame(dicts)
    return df

if __name__ == '__main__':
    # example of loading history
    hist = load_logs('gs://preetum/logs/cifar-test-2/q5fy5rq1/history')
