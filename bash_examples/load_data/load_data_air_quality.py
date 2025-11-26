import os
import sys
import torch
import logging
import numpy as np
import pandas as pd
from datetime import timedelta
import matplotlib.pyplot as plt
from dsipts import TimeSeries, RNN, read_public_dataset, LinearTS, Persistent
import shap
import torch
import numpy as np
import pandas as pd


# Global parameters
past_steps = 72
future_steps = 24
use_covariates = True  #False if use only y, True to use also covariates
use_future_covariate = True # suppose to have some future covariates

# Enable TF32 for matmul and cudnn
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

# Splitting parameters
split_params = {
    'perc_train': 0.7,
    'perc_valid': 0.1,
    'range_train': None,
    'range_validation': None,
    'range_test': None,
    'past_steps': past_steps,
    'future_steps': future_steps,
    'starting_point': None,
    'skip_step': 1  # I don't know what this is
}

DIR_PATH = "/Users/federicorubbi/Documents/unitn/public-ai-challenge/appa-chinquinaria/data/merged_all_datasets.csv"

# logger = logging.getLogger()
# logger.setLevel(logging.DEBUG)
# logging.debug("info")

def process_data(df):
    df.rename(columns = ({'datetime':'time'}), inplace=True)
    df.time = pd.to_datetime(df.time, format='%Y-%m-%d %H:%M:%S')
    return df


def load_to_ts(df, conf):
    ts = TimeSeries(conf.ts.name)

    df["hour"] = df["time"].dt.hour.astype(float)
    df["month"] = df["time"].dt.month.astype(float)
    df["day"] = df["time"].dt.day.astype(float)
    df["weekday"] = df["time"].dt.weekday.astype(float)

    # Extract past variables
    past_variables = df.columns.drop(['time']) if conf.ts.get('use_covariates',False) else []  # all columns except time
    print("Past variables used:", past_variables)
    print(type(past_variables))
    print("df columns types:", df.dtypes)
    print("numbers of nan in date column:", df['time'].isna().sum())

    # Extract all target variables
    target_variables = [c for c in df.columns if c.startswith('pm10')]
    print("Target variables used:", target_variables)
    print(type(target_variables))
    print("numbers of nan in target variables:", df[target_variables].isna().sum())

    # Extract future variables
    use_future_covariates = conf.ts.get('use_future_covariates',False)
    if use_future_covariates:
        future_variables = df.columns.drop(['time'])
        future_variables = future_variables.drop(target_variables)
        print("Future variables used:", future_variables)
        print(type(future_variables))
        print("numbers of nan in date column:", df['time'].isna().sum())
    else:
        future_variables = []
        print("No future variables used")
        print(type(future_variables))

    ts = TimeSeries(conf.ts.name)
    ts.load_signal(
        df, enrich_cat= conf.ts.get('enrich',[]),
        target_variables=target_variables,  # PM10 values
        past_variables=list(past_variables),
        future_variables=list(future_variables),  # Should I remove "hour", "month", "day", "weekday" from future variables?
        cat_past_var=[],
        cat_fut_var=[],
        silly_model=conf.ts.get('silly',False)
    )

    # Optional: print how many training/validation/test samples will be produced
    try:
        dl_train, dl_val, dl_test = ts.split_for_train(**split_params)
        print("Generated samples (train/val/test):",
              len(dl_train.dataset) if dl_train is not None else 0,
              len(dl_val.dataset) if dl_val is not None else 0,
              len(dl_test.dataset) if dl_test is not None else 0)
    except Exception as e:
        print("Could not compute split sizes:", e)

    return ts


def load_data(conf):
    # get current folder
    current_folder = os.getcwd()
    data_path = conf.dataset.path
    # read a dataframe
    df = pd.read_csv(data_path, low_memory=False)
    # process the dataframe
    df = process_data(df)
    print("Dataframe processed.")
    print(df.head())
    # load the timeseries to the datastructure, adding the hour column and use all the covariates
    ts = load_to_ts(df, conf)
    return ts
