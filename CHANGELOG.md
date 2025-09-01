## 1.1.5 (2025-08-29)
- rewriting most of the modules for handling different future and past categorical variables
- extension of categorical and future covariates in almost all the models
- `uv` full management of the package
- refactoring almost all the structure and documentation

## 1.1.4 (2025-08-22)
- added `restart: true` tro model configuration to restart the training procedure: carefurl the max_epochs should be increased if you need to retrain

## 1.1.4 (2025-07-29)
- bug fixing tuner learning rate
- added TTM model and TimeXer
- added compatibility with newer version of lightening and torch

## 1.1.1 
- added [SAM optimizer](https://arxiv.org/pdf/2402.10198) 
```bash
 python train.py  --config-dir=config_test --config-name=config architecture=itransformer dataset.path=/home/agobbi/Projects/ExpTS/data train_config.dirpath=tmp inference=tmp model_configs.optim=SAM +optim_config.rho=0.5
 ```