# aer-analysis

Replication files for AER.

> L. Wong, D. Liu, L. Berti-Equille, and K. Veeramachaneni. "Time Series Anomaly Detection using Prediction-Reconstruction Mixture Errors" IEEE BigData 2022. 

## Usage

Experiments were made in **python 3.7**.
To reproduce the analysis made, create a virtual environment and install required packages.

```bash
conda create --name aer-env python=3.7
conda activate aer-env
pip install -r requirements.txt
```

\[Optional\] If you want to run benchmark scripts, please download data folder.
- download [data-aer-paper.zip](https://d3-ai-orion.s3.amazonaws.com/data-aer-paper.zip)
- unzip it into `aer/data`

To reproduce tables and figures, use `notebooks/analyze_results.ipynb`

## Resources

* Anomaly detection in time series using [Orion](https://github.com/sintel-dev/Orion)
