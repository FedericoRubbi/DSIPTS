**Installation guide**


```bash
uv venv --python 3.11
source .venv/bin/activate
```

##old torch and pytorch-lightning
```bash
uv pip install 'numpy<2.0.0' matplotlib pandas scikit-learn 'torch==2.4.1' 'pytorch-lightning==1.9.4' hydra-core hydra-joblib-launcher hydra-optuna-sweeper 'beautifulsoup4==4.12.0' html5lib 'html-table-parser-python3==0.3.1' sphinx plotly scikit-learn numba einops sphinx_pdj_theme sphinx_mdinclude 'aim==3.20.1' transformers 
```

##new torch and pytorch-lightning
```bash
uv pip install 'numpy<2.0.0' matplotlib pandas scikit-learn lightning hydra-core hydra-joblib-launcher hydra-optuna-sweeper 'beautifulsoup4==4.12.0' html5lib 'html-table-parser-python3==0.3.1' sphinx plotly scikit-learn numba einops sphinx_pdj_theme sphinx_mdinclude 'aim==3.20.1' transformers 
python setup_uv.py install --force
```

```bash
python setup_uv.py install --force
```


Minimal example for training:
```bash
python train.py  --config-dir=config_test --config-name=config architecture=itransformer,timexer dataset.path=/home/agobbi/Projects/ExpTS/data model_configs.past_steps=96 split_params.skip_step=30 -m
```

and comparison 
```bash
 python compare.py  --config-dir=config_test --config-name=compare +inference.output_path=tmp
 ```