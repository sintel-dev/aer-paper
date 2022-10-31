#!/usr/bin/env python3

"""
Generate and analyze the results for the paper: "Time Series 
Anomaly Detection using Prediction-Reconstruction Mixture Errors"
"""

import os
import sys
import logging
import warnings
import pandas as pd
from functools import partial

from aer.benchmark import benchmark, BENCHMARK_DATA, METRICS
from orion.evaluation import contextual_confusion_matrix
from orion.evaluation.contextual import record_observed, record_expected
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style("whitegrid")

warnings.simplefilter('ignore')

LOGGER = logging.getLogger(__name__)

# Datasets
NAB = ['artificialWithAnomaly', 'realAdExchange', 'realAWSCloudwatch', 'realTraffic', 'realTweets']
NASA = ['MSL', 'SMAP']
YAHOO = ['YAHOOA1', 'YAHOOA2', 'YAHOOA3', 'YAHOOA4']
UCR = ['UCR']
ALL_DATASETS = NAB + NASA + YAHOO + UCR

FAMILY = {
    "MSL": "NASA",
    "SMAP": "NASA",
    "YAHOOA1": "YAHOO",
    "YAHOOA2": "YAHOO",
    "YAHOOA3": "YAHOO",
    "YAHOOA4": "YAHOO",
    "artificialWithAnomaly": "NAB",
    "realAWSCloudwatch": "NAB",
    "realAdExchange": "NAB",
    "realTraffic": "NAB",
    "realTweets": "NAB",
    "UCR": "UCR"
}

DATASET_RENAMES = {
    "MSL": "MSL", 
    "SMAP": "SMAP", 
    "YAHOOA1": "A1", 
    "YAHOOA2": "A2", 
    "YAHOOA3": "A3", 
    "YAHOOA4": "A4",
    "artificialWithAnomaly": "Art", 
    "realAWSCloudwatch": "AdEx", 
    "realAdExchange": "AWS", 
    "realTraffic": "Traffic", 
    "realTweets": "Tweets", 
    "UCR": "UCR" 
}

PIPELINE_TO_COLOR_MAP = {
    'ARIMA': '#FAA31B',
    'LSTM-DT': '#88C6ED',
    'LSTM-AE': '#82C341',
    'LSTM-VAE': '#D54799',
}

PREDICTION_BASED_MODELS = ['ARIMA', 'LSTM-DT']
RECONSTRUCTION_BASED_MODELS = ['LSTM-AE', 'LSTM-VAE']
REC_ERROR_TYPES = ['point', 'area', 'dtw']

# Path to save experiment results and logs
RESULTS_DIRECTORY = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'results')
LOGS_DIRECTORY = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'logs')
MODELS_DIRECTORY = os.path.join(RESULTS_DIRECTORY, 'models')
os.makedirs(RESULTS_DIRECTORY, exist_ok = True)
os.makedirs(LOGS_DIRECTORY, exist_ok = True)

# Additional Metrics
del METRICS['accuracy']
METRICS['confusion_matrix'] = contextual_confusion_matrix
METRICS['observed'] = record_observed
METRICS['expected'] = record_expected
METRICS = {k: partial(fun, weighted=False) for k, fun in METRICS.items()}


# ------------------------------------------------------------------------------
# Running pipelines to generate results necessary for analysis
# ------------------------------------------------------------------------------
def _run_experiment(experiment_name: str, pipelines: dict, datasets: list, metrics: dict,
                   results_directory: str = RESULTS_DIRECTORY, workers: int = 1,
                   tqdm_log_file: str = 'output.txt'):
    datasets = {key: BENCHMARK_DATA[key] for key in datasets}
    scores = benchmark(
        pipelines=pipelines,
        datasets=datasets,
        metrics=metrics,
        rank='f1',
        show_progress=True,
        workers=workers,
        # cache_dir=os.path.join(results_directory, experiment_name, 'cache'),
        # pipeline_dir=os.path.join(results_directory, experiment_name, 'pipeline'),
        tqdm_log_file=tqdm_log_file
    )
    return scores
        
def run_table_IV_A_nomask():
    experiment_name="Table_IV_A_no-mask"
    pipelines = {
        'arima': 'arima_ablation',
        'lstm_dynamic_threshold': 'lstm_dynamic_threshold_ablation',
        'lstm_autoencoder': 'lstm_autoencoder_ablation',
        'vae': 'vae_ablation',
        'tadgan': 'tadgan_ablation'
    }
    key_maps = {
        'arima': 'ARIMA',
        'lstm_dynamic_threshold': 'LSTM-DT',
        'lstm_autoencoder': 'LSTM-AE',
        'vae': 'LSTM-VAE',
        'tadgan': 'TadGAN'

    } 
    for key in pipelines:
        _results = _run_experiment(
            experiment_name=experiment_name,
            pipelines={key: pipelines[key]},
            datasets=ALL_DATASETS,
            metrics=METRICS,
            results_directory=RESULTS_DIRECTORY,
            workers=1,
            tqdm_log_file = f'{LOGS_DIRECTORY}/{key_maps[key]}.txt'
        )
        _results['pipeline'] = key_maps[key]
        _results.to_csv(f'{RESULTS_DIRECTORY}/{key_maps[key]}_results.csv', index=False)

def run_table_IV_A_mask():
    experiment_name="Table_IV_A_mask"
    pipelines = {
        'arima': 'arima_ablation-mask',
        'lstm_dynamic_threshold': 'lstm_dynamic_threshold_ablation-mask',
        'bi_reg': 'bi_reg_ablation',   # this pipeline comes with mask naturally
        'lstm_autoencoder': 'lstm_autoencoder_ablation-mask',
        'vae': 'vae_ablation-mask',
        'tadgan': 'tadgan_ablation-mask'
    }
    key_maps = {
        'arima': 'ARIMA (M)',
        'lstm_dynamic_threshold': 'LSTM-DT (M)',
        'bi_reg': 'LSTM-DT (M, Bi)',
        'lstm_autoencoder': 'LSTM-AE (M)',
        'vae': 'LSTM-VAE (M)',
        'tadgan': 'TadGAN (M)'

    } 
    for key in pipelines:
        _results = _run_experiment(
            experiment_name=experiment_name,
            pipelines={key: pipelines[key]},
            datasets=ALL_DATASETS,
            metrics=METRICS,
            results_directory=RESULTS_DIRECTORY,
            workers=1,
            tqdm_log_file = f'{LOGS_DIRECTORY}/{key_maps[key]}.txt'
        )
        _results['pipeline'] = key_maps[key]
        _results.to_csv(f'{RESULTS_DIRECTORY}/{key_maps[key]}_results.csv', index=False)

def run_table_IV_A():
    run_table_IV_A_nomask()    
    run_table_IV_A_mask()
    
def run_table_IV_B():
    experiment_name="Table_IV_B"
    pipelines = ['aer_ablation-mult', 'aer_ablation-sum', 'aer_ablation-pred', 'aer_ablation-rec']
    key_maps = {
        'aer_ablation-mult': 'AER (MULT)',
        'aer_ablation-sum': 'AER (SUM)',
        'aer_ablation-pred': 'AER (PRED)',
        'aer_ablation-rec': 'AER (REC)'
    } 
    for key in pipelines:
        _results = _run_experiment(
            experiment_name=experiment_name,
            pipelines={'aer': key},
            datasets=ALL_DATASETS,
            metrics=METRICS,
            results_directory=RESULTS_DIRECTORY,
            workers=1,
            tqdm_log_file = f'{LOGS_DIRECTORY}/{key_maps[key]}.txt'
        )
        _results['pipeline'] = key_maps[key]
        _results.to_csv(f'{RESULTS_DIRECTORY}/{key_maps[key]}_results.csv', index=False)

# ------------------------------------------------------------------------------
# Analyzing results
# ------------------------------------------------------------------------------
def _get_table_summary(result_files, results_path):
    results = None
    for filename in result_files:
        result = pd.read_csv(f"{results_path}/{filename}_results.csv")
        result['pipeline'] = filename  # todo: decide whether to keep it or not
        if results is None:
            results = result
        else:
            results = pd.concat([results, result])
    
    order_pipelines = result_files 
    order_datasets = DATASET_RENAMES.values()
    
    df = results.copy(deep=True)
    df['group'] = df['dataset'].apply(FAMILY.get)
    df['dataset'] = df['dataset'].apply(DATASET_RENAMES.get)

    df = df.groupby(['group', 'dataset', 'pipeline'])[['fp', 'fn', 'tp']].sum().reset_index()
    df['precision'] = df.eval('tp / (tp + fp)')
    df['recall'] = df.eval('tp / (tp + fn)')
    df['f1'] = df.eval('2 * (precision * recall) / (precision + recall)')

    df = df.set_index(['dataset', 'pipeline'])['f1'].unstack(0)
    df = df[order_datasets]
    df['AVG (F1)'] = df.mean(axis=1)
    df['SD (F1)'] = df.std(axis=1)

    return df.T[order_pipelines].T
        
def analyze_table_IV_A(results_path=RESULTS_DIRECTORY):
    result_files = ['ARIMA', 'ARIMA (M)', 'LSTM-DT', 
                    'LSTM-DT (M)', 'LSTM-DT (M, Bi)',
                    'LSTM-AE', 'LSTM-AE (M)', 'LSTM-VAE',
                    'LSTM-VAE (M)', 'TadGAN', 'TadGAN (M)'] 
    return _get_table_summary(result_files, results_path)

def analyze_table_IV_B(results_path=RESULTS_DIRECTORY):
    result_files = ['AER (PRED)', 'AER (SUM)', 'AER (REC)', 'AER (MULT)']
    return _get_table_summary(result_files, results_path)
    
def analyze_table_III(results_path=RESULTS_DIRECTORY):
    aer_result_files = ['AER (MULT)', 'AER (PRED)']
    # A3,A4 use (PRED) and others use (MULT)
    df1 = _get_table_summary(aer_result_files, results_path)
    df1.loc['AER (MULT)']['A3'] = df1.loc['AER (PRED)']['A3']
    df1.loc['AER (MULT)']['A4'] = df1.loc['AER (PRED)']['A4']
    # re-calculate mean/std
    columns = DATASET_RENAMES.values()
    df1.loc['AER (MULT)']['AVG (F1)'] = df1.loc['AER (MULT)'][columns].mean()
    df1.loc['AER (MULT)']['SD (F1)'] = df1.loc['AER (MULT)'][columns].std()
    
    other_result_files = ['ARIMA', 'LSTM-DT', 'LSTM-AE', 'LSTM-VAE', 'TadGAN']
    df2 = _get_table_summary(other_result_files, results_path)
    
    df2.loc['AER'] = df1.loc['AER (MULT)']
    
    return df2

# ------------------------------------------------------------------------------
# Plotting benchmark
# ------------------------------------------------------------------------------

def plot_anomaly_scores(dataset: str, signal_name: str) -> None:
    fig, axs = plt.subplots(5, 1, figsize=(20, 25), sharex=True)

    # Graph (a): Signal and Anomalies
    signal = pd.read_csv(os.path.join(MODELS_DIRECTORY, signal_name, 'signal.csv'))
    axs[0].plot(signal['timestamp'], signal['value'], color='#050505')
    expected = pd.read_csv(os.path.join(MODELS_DIRECTORY, signal_name, 'anomalies.csv'))
    for start, end in zip(expected['start'], expected['end']):
        axs[0].axvspan(start-1, end+1, color='#FF0000', alpha=0.5)
    axs[0].set_title(f"Anomalies for {signal_name} from {dataset}", fontsize=24)

    # Graph (b): Prediction-based Anomaly Scores
    for model_name in PREDICTION_BASED_MODELS:
        model_predictions = pd.read_csv(os.path.join(MODELS_DIRECTORY, signal_name, '{}.csv'.format(model_name)))
        axs[1].plot(model_predictions['index'], model_predictions['errors'], color=PIPELINE_TO_COLOR_MAP[model_name], label=model_name)

    axs[1].set_title("Prediction-based Anomaly Scores", fontsize=24)
    axs[1].legend(loc='upper right', ncol=len(PREDICTION_BASED_MODELS), prop={'size': 18})
    axs[1].axes.xaxis.set_ticklabels([])

    # Graph (c-e): Reconstruction-based Anomaly Scores
    for idx, rec_error_type in enumerate(REC_ERROR_TYPES):
        for model_name in RECONSTRUCTION_BASED_MODELS:
            model_predictions = pd.read_csv(os.path.join(MODELS_DIRECTORY, signal_name, '{}_{}.csv'.format(model_name, rec_error_type.upper())))
            axs[2+idx].plot(model_predictions['index'], model_predictions['errors'], color=PIPELINE_TO_COLOR_MAP[model_name], label=model_name)

        axs[2+idx].set_title(f"Reconstruction-based Anomaly Scores ({rec_error_type.upper()})", fontsize=24)
        axs[2+idx].legend(loc='upper right', ncol=len(RECONSTRUCTION_BASED_MODELS), prop={'size': 18})

    plt.rcParams.update({'font.size': 18})
    plt.show()


def make_figure_3():
    dataset = 'artificialWithAnomaly'
    signal_name = 'art_daily_flatmiddle'
    plot_anomaly_scores(dataset, signal_name)

def make_figure_4():
    dataset = 'YAHOOA3'
    signal_name = 'A3Benchmark-TS11'
    plot_anomaly_scores(dataset, signal_name)

def make_figure_6():
    # todo: lawrence
    # can you take the code here and write a version for our figure
    # https://github.com/sarahmish/sintel-paper/blob/master/analysis.py#L171
    pass

if __name__ == '__main__':
    print('running pipelines in Table IV-A')
    run_table_IV_A()
    print('running pipelines in Table IV-B')
    run_table_IV_B()