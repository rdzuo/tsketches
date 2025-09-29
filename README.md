# tsketches
The offcial code of Tsketches. The Appendix of paper is provided in Paper_with_appendix.pdf.

## Setup
```bash
pip install -r requirements.txt
```
The datasets are the top-8 longest time series datates from www.timeseriesclassification.com.
Users can download the data after z-normlization and tsketches of all datasets from https://shorturl.at/a5PnP.

Create data folder
```bash
mkdir data
```
Put a specific dataset (e.g., BinaryHeartbeat) as follows:

```bash
mkdir data/raw/BinaryHeartbeat
mkdir data/tsketch/BinaryHeartbeat
```

## Run an example
To run an example:
```bash
python main.py --dataset BinaryHeartbeat
```

## For other datasets and models

Using tsketches to train a dataset with name "dataset_name"
```bash
python main.py  --dataset dataset_name
```
Using tsketches to train a dataset with name "dataset_name" and model with name "model_name"

## Prototypes
We have If you would like to obtain prototypes and tsketches by your own dataset.
```bash
python autoencoder.py  --dataset dataset_name
```

## Extend to other transformer

If you would like to use tsketches to train your own transformer models, put the codes of your model to models.py.
