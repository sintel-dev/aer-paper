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
from IPython.display import display
import matplotlib.pyplot as plt
import seaborn as sns

warnings.simplefilter('ignore')

LOGGER = logging.getLogger(__name__)

# Datasets
NAB = ['artificialWithAnomaly', 'realAdExchange', 'realAWSCloudwatch', 'realTraffic', 'realTweets']
NASA = ['MSL', 'SMAP']
YAHOO = ['YAHOOA1', 'YAHOOA2', 'YAHOOA3', 'YAHOOA4']
UCR = ['UCR']
ALL_DATASETS = NAB + NASA + YAHOO + UCR
ALL_DATASETS_EXCLUDING_YAHOO = NAB + NASA + UCR # by default running

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
    'ARIMA': '#d37d0b',
    'LSTM-DT': '#83d6ff',
    'LSTM-AE': '#64aa13',
    'LSTM-VAE': '#9612b2',
}

PREDICTION_BASED_MODELS = ['ARIMA', 'LSTM-DT']
RECONSTRUCTION_BASED_MODELS = ['LSTM-AE', 'LSTM-VAE']
REC_ERROR_TYPES = ['point', 'area', 'dtw']
MODELS = ['ARIMA', 'LSTM-DT', 'LSTM-AE', 'LSTM-VAE', 'TadGAN', 'AER']

# Path to save experiment results and logs
RESULTS_DIRECTORY = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'results')
LOGS_DIRECTORY = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'logs')
MODELS_DIRECTORY = os.path.join(RESULTS_DIRECTORY, 'models')
PAPER_RESULTS_DIRECTORY = os.path.join(RESULTS_DIRECTORY, 'paper-results')
FIGURES_DIRECTORY = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'figures')
os.makedirs(RESULTS_DIRECTORY, exist_ok=True)
os.makedirs(LOGS_DIRECTORY, exist_ok=True)
os.makedirs(FIGURES_DIRECTORY, exist_ok = True)

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
        # the following two parameters are used for saving the 
        # intermediate prediction results and trained models
        # cache_dir=os.path.join(results_directory, experiment_name, 'cache'),
        # pipeline_dir=os.path.join(results_directory, experiment_name, 'pipeline'),
        tqdm_log_file=tqdm_log_file
    )
    return scores


def run_table_IV_A_nomask():
    experiment_name = "Table_IV_A_no-mask"
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
            datasets=ALL_DATASETS_EXCLUDING_YAHOO,
            metrics=METRICS,
            results_directory=RESULTS_DIRECTORY,
            workers=1,
            tqdm_log_file=f'{LOGS_DIRECTORY}/{key_maps[key]}.txt'
        )
        _results['pipeline'] = key_maps[key]
        _results.to_csv(f'{RESULTS_DIRECTORY}/{key_maps[key]}_results.csv', index=False)


def run_table_IV_A_mask():
    experiment_name = "Table_IV_A_mask"
    pipelines = {
        'arima': 'arima_ablation-mask',
        'lstm_dynamic_threshold': 'lstm_dynamic_threshold_ablation-mask',
        'bi_reg': 'bi_reg_ablation',  # this pipeline comes with mask naturally
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
            datasets=ALL_DATASETS_EXCLUDING_YAHOO,
            metrics=METRICS,
            results_directory=RESULTS_DIRECTORY,
            workers=1,
            tqdm_log_file=f'{LOGS_DIRECTORY}/{key_maps[key]}.txt'
        )
        _results['pipeline'] = key_maps[key]
        _results.to_csv(f'{RESULTS_DIRECTORY}/{key_maps[key]}_results.csv', index=False)


def run_table_IV_A():
    run_table_IV_A_nomask()
    run_table_IV_A_mask()


def run_table_IV_B():
    experiment_name = "Table_IV_B"
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
            datasets=ALL_DATASETS_EXCLUDING_YAHOO,
            metrics=METRICS,
            results_directory=RESULTS_DIRECTORY,
            workers=1,
            tqdm_log_file=f'{LOGS_DIRECTORY}/{key_maps[key]}.txt'
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
    df['SD (F1)'] = df.std(axis=1).map(lambda x: f'{x:.2f}')

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
    df1.loc['AER (MULT)']['SD (F1)'] = f"{df1.loc['AER (MULT)'][columns].std():.2f}"

    other_result_files = ['ARIMA', 'LSTM-DT', 'LSTM-AE', 'LSTM-VAE', 'TadGAN']
    df2 = _get_table_summary(other_result_files, results_path)

    df2.loc['AER'] = df1.loc['AER (MULT)']

    return df2

# ------------------------------------------------------------------------------
# Saving results
# ------------------------------------------------------------------------------

def _savefig(fig, name, figdir=FIGURES_DIRECTORY):
    # for ext in ['.png', '.pdf', '.eps', '.svg']:
    for ext in ['.png']:
        fig.savefig(f'{figdir}/{name}{ext}',
                    bbox_inches='tight')

# ------------------------------------------------------------------------------
# Plotting benchmark
# ------------------------------------------------------------------------------

def plot_anomaly_scores(dataset: str, signal_name: str) -> None:
    
    sns.set_theme(context='paper', style='whitegrid', font_scale=1.6)
    fig, axs = plt.subplots(5, 1, figsize=(20, 25), sharex=True)

    # Graph (a): Signal and Anomalies
    signal = pd.read_csv(os.path.join(MODELS_DIRECTORY, signal_name, 'signal.csv'))
    axs[0].plot(signal['timestamp'], signal['value'], color='#5d7793')
    expected = pd.read_csv(os.path.join(MODELS_DIRECTORY, signal_name, 'anomalies.csv'))
    for start, end in zip(expected['start'], expected['end']):
        axs[0].axvspan(start - 1, end + 1, color='#FF0000', alpha=0.5)
    axs[0].set_title(f"Anomalies for {signal_name} from {dataset}", fontsize=24)

    # Graph (b): Prediction-based Anomaly Scores
    for model_name in PREDICTION_BASED_MODELS:
        model_predictions = pd.read_csv(os.path.join(MODELS_DIRECTORY, signal_name, '{}.csv'.format(model_name)))
        axs[1].plot(model_predictions['index'], model_predictions['errors'], color=PIPELINE_TO_COLOR_MAP[model_name],
                    label=model_name)

    axs[1].set_title("Prediction-based Anomaly Scores", fontsize=24)
    axs[1].legend(loc='upper right', ncol=len(PREDICTION_BASED_MODELS), prop={'size': 18})
    axs[1].axes.xaxis.set_ticklabels([])

    # Graph (c-e): Reconstruction-based Anomaly Scores
    for idx, rec_error_type in enumerate(REC_ERROR_TYPES):
        for model_name in RECONSTRUCTION_BASED_MODELS:
            model_predictions = pd.read_csv(
                os.path.join(MODELS_DIRECTORY, signal_name, '{}_{}.csv'.format(model_name, rec_error_type.upper())))
            axs[2 + idx].plot(model_predictions['index'], model_predictions['errors'],
                              color=PIPELINE_TO_COLOR_MAP[model_name], label=model_name)

        axs[2 + idx].set_title(f"Reconstruction-based Anomaly Scores ({rec_error_type.upper()})", fontsize=24)
        axs[2 + idx].legend(loc='upper right', ncol=len(RECONSTRUCTION_BASED_MODELS), prop={'size': 18})

    plt.rcParams.update({'font.size': 18})
    plt.show()
    return fig

def make_figure_3():
    dataset = 'artificialWithAnomaly'
    signal_name = 'art_daily_flatmiddle'
    fig = plot_anomaly_scores(dataset, signal_name)
    _savefig(fig, 'figure3')

def make_figure_4():
    dataset = 'YAHOOA3'
    signal_name = 'A3Benchmark-TS11'
    fig = plot_anomaly_scores(dataset, signal_name)
    _savefig(fig, 'figure4')

def make_figure_6(show_numerical_results: bool = False):
    # View Numerical Results
    signals_to_size_map = {
        '140-InternalBleeding4': 20,
        '192-s20101mML2': 200,
        '234-mit14157longtermecg': 2000
    }
    load_results = lambda model: pd.read_csv(os.path.join(PAPER_RESULTS_DIRECTORY, model + '_results.csv'))
    runtime_results = dict()

    for model in MODELS:
        model_results = load_results(model + ' (MULT)' if model == 'AER' else model)
        for signal in signals_to_size_map.keys():
            runtime_results.setdefault(model, [])
            elapsed = round(model_results[model_results.signal == signal]['elapsed'].iloc[0])
            runtime_results[model].append(elapsed)

    runtime_results = pd.DataFrame(runtime_results)
    runtime_results.index = signals_to_size_map
    runtime_results = runtime_results.T

    if show_numerical_results:
        display(runtime_results)

    # Construct and plot graph
    runtime_results_graph = []
    for signal, size in signals_to_size_map.items():
        for model_name in MODELS:
            runtime_results_graph.append(
                ['{}\n({})'.format(size, signal), runtime_results[signal].loc[model_name], model_name])

    runtime_results_graph = pd.DataFrame(runtime_results_graph, columns=['Signal', 'Seconds', 'Model'])

    # ARIMA, LSTM-DT, LSTM-AE, LSTM-VAE, TadGAN, AER
    _COLORS = ["#d37d0b", "#83d6ff", "#64aa13", "#9612b2", "#273643", "#f1b145"]
    _PALETTE = sns.color_palette(_COLORS)

    sns.set_theme(context='paper', style='whitegrid', font_scale=1.4)
    fig = plt.figure(figsize=(11, 5))
    ax = sns.barplot(data=runtime_results_graph, x='Signal', y='Seconds', hue='Model', palette=_PALETTE, saturation=0.7,
                     linewidth=0.5, edgecolor='k')
    ax.set(yscale='log')
    plt.xlabel('Signal Size (kb) and Name', fontsize=14, labelpad=10, fontweight='bold')
    plt.ylabel('Total Execution Time in Seconds (log)', fontsize=14, fontweight='bold')
    ax.grid(True, linestyle='--')
    ax.legend(loc='lower center', bbox_to_anchor=(0.5, 1), ncol=len(MODELS), fancybox=True, shadow=True)
    plt.show()
    _savefig(fig, 'figure6')


if __name__ == '__main__':
    print('running pipelines in Table IV-A')
    run_table_IV_A()
    print('running pipelines in Table IV-B')
    run_table_IV_B()
