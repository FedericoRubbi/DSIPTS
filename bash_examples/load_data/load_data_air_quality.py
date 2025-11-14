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
past_steps = 4
future_steps = 8
use_covariates = True  #False if use only y, True to use also covariates
use_future_covariate = False # suppose to have some future covariates

# Enable TF32 for matmul and cudnn
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

# Splitting parameters
split_params = {
    'perc_train': 0.8,
    'perc_valid': 0.1,
    'range_train': None,
    'range_validation': None,
    'range_test': None,
    'past_steps': past_steps,
    'future_steps': future_steps,
    'starting_point': None,
    'skip_step': 1
}

DIR_PATH = "./data/merged_appa_eea_by_proximity_v2.csv"

# logger = logging.getLogger()
# logger.setLevel(logging.DEBUG)
# logging.debug("info")

# "SPO.IT1619A_5_BETA_2004-05-04_00:00:00_Valore,"
# "SPO.IT1619A_5_BETA_2004-05-04_00:00:00_Latitudine,"
# "SPO.IT1619A_5_BETA_2004-05-04_00:00:00_Longitudine,"
# "SPO.IT1619A_5_BETA_2004-05-04_00:00:00_Inquinante,"
# "SPO.IT1619A_5_BETA_2004-05-04_00:00:00_Unità_misura"

def process_data(df):
    keep_col = df.columns[
        (df.columns.str.endswith('Latitudine')==False) & 
        (df.columns.str.endswith('Longitudine')==False) & 
        (df.columns.str.endswith('Inquinante')==False) & 
        (df.columns.str.endswith('misura')==False) 
        # & (df.columns.str.contains('SPO')==False)
    ]
    df = df[keep_col]
    df.drop([
        'Nazione',
        'Comune',
        'StazioneMeteo',
        # 'Latitudine',
        # 'Longitudine',
        # 'Inquinante'
    ], axis=1, inplace=True)
    df.rename(columns = ({'Data':'time'}), inplace=True)
    df.time = pd.to_datetime(df.time, format='%Y-%m-%d')
    variables = df.columns.drop(['Stazione'])
    return df


def train_RNN(ts):
    #train the model for 50 epochs with auto_lr_find 
    ts.train_model(
        dirpath="./models/RNN",
        split_params=split_params,
        batch_size=128,
        num_workers=4,
        max_epochs=5,
        gradient_clip_val=0.0,
        gradient_clip_algorithm='value',
        precision='bf16-mixed',
        auto_lr_find=True
    )
    # Print the losses, check overfitting
    ts.save("PM10_RNN")


def test_RNN(ts):
    """Make inferences on the test set."""
    res = ts.inference_on_set(200, 4, set='test', rescaling=True)
    res.head()
    return res


def plot_results(res, station=None, lag=8):
    """Plot prediction results with confidence bands."""
    plt.figure(figsize=(15, 7))
    
    # Filter data for better visualization
    to_plot = res[res.time > pd.to_datetime('2020-12-31')]
    
    if station:
        to_plot = to_plot[to_plot.Stazione == station]
    lag_data = to_plot[to_plot.lag == lag]
    
    average_bias = 0.0
    # average_bias = (lag_data.Valore_median - lag_data.Valore).abs().mean()
    # print(f'Average absolute bias for lag={lag}: {average_bias:.4f}')

    # Plot actual vs predicted values
    plt.plot(lag_data.time, lag_data.Valore, label='real', alpha=0.5)
    plt.plot(lag_data.prediction_time, lag_data.Valore_median-average_bias, label='median', alpha=0.5)
    plt.fill_between(lag_data.prediction_time, lag_data.Valore_low-average_bias, lag_data.Valore_high-average_bias, 
                     alpha=0.2, label='error band')
    plt.title(f'Prediction on test for lag={lag}')
    plt.legend()
    plt.show()


def get_rnn_config(ts):
    config = dict(
        model_configs=dict(
            # Task dependent
            past_steps=past_steps,
            future_steps=future_steps,
            
            # Categorical embeddings
            emb_dim=8,
            use_classical_positional_encoder=True,
            reduction_mode='mean',
            
            # Model architecture
            kind='gru',
            hidden_RNN=12,
            num_layers_RNN=2,
            kernel_size=15,
            dropout_rate=0.5,
            remove_last=True,
            use_bn=False,
            activation='torch.nn.PReLU',
            
            # Loss configuration
            quantiles=[0.1, 0.5, 0.9],
            persistence_weight=0.010,
            loss_type='l1',
            
            # Optimizer
            optim='torch.optim.Adam',
            
            # Dataset dependent parameters
            past_channels=len(ts.past_variables),
            future_channels=len(ts.future_variables),
            embs_past=[ts.dataset[c].nunique() for c in ts.cat_past_var],
            embs_fut=[ts.dataset[c].nunique() for c in ts.cat_fut_var],
            out_channels=len(ts.target_variables)
        ),
        scheduler_config=dict(gamma=0.1, step_size=100),
        optim_config=dict(lr=0.0005, weight_decay=0.01)
    )
    return config


def fit_RNN(ts):
    # Config dictionary
    config = get_rnn_config(ts)
    model_rnn = RNN(
        **config['model_configs'],
        optim_config=config['optim_config'],
        scheduler_config=config['scheduler_config'],
        verbose=True
    )
    #set the desirere model
    ts.set_model(model_rnn,config=config)
    print(str(ts))
    #train the model
    train_RNN(ts) 
    #test the model
    res= test_RNN(ts)

    print(res.head())
    # result diagnostic
    plot_results(res, station='Parco S. Chiara')
    # plot_results(res, station='Monte Gaza')


def check_cuda():
    if torch.cuda.is_available():
        print("CUDA is available. Device count:", torch.cuda.device_count())
        for device in range(torch.cuda.device_count()):
            print(f"Device {device}: {torch.cuda.get_device_name(device)}")
    else:
        print("CUDA is not available. Please check your installation.")


def load_to_ts(df, conf):
    ts = TimeSeries(conf.ts.name)
    df["Hour"] = df["time"].dt.hour.astype(float)
    df["Month"] = df["time"].dt.month.astype(float)
    df["Day"] = df["time"].dt.day.astype(float)
    df["Weekday"] = df["time"].dt.weekday.astype(float)

    # Exclude categorical columns from numerical variables
    numerical_cols = df.columns.drop(['Stazione','time']) if conf.ts.get('use_covariates',False) else []
    print("Numerical columns used:", numerical_cols)
    print(type(numerical_cols))
    print("df columns types:", df.dtypes)
    print("numbers of nan in date column:", df['time'].isna().sum())

    ts = TimeSeries(conf.ts.name)
    ts.load_signal(
        df, enrich_cat= conf.ts.get('enrich',[]),
        target_variables=['Valore'],  # PM10 value
        past_variables=list(numerical_cols),
        # future_variables=numerical_cols if use_future_covariate else [],
        future_variables=[],
        cat_past_var=['Stazione'],
        cat_fut_var=['Stazione'], 
        group='Stazione',
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




    
