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

To reproduce tables and figures, use `notebooks/analyze_results.ipynb`

\[Optional\] If you want to run benchmark scripts, please download data folder.
- download [data-aer-paper.zip](https://sintel-orion.s3.amazonaws.com/data-aer-paper.zip)
- unzip it into `aer/data`

Note: [Yahoo datasets](https://webscope.sandbox.yahoo.com/catalog.php?datatype=s&did=70&guccounter=1) are excluded, to include which you may need to apply for the access on the official website and then add those signals to data folder.

Now you should be able to use `notebooks/run_benchmark.ipynb` to benchmark all pipelines and their variations. The results will be saved into the folder `./results`. You can check the running progress of each pipeline in `./logs`. 
> :warning: It will take a very long time to run, if you run all pipelines on all datasets. You may want to customize it by selecting a subset of pipelines or datasets in `analysis.py`. 


## Resources

* Anomaly detection in time series using [Orion](https://github.com/sintel-dev/Orion)
