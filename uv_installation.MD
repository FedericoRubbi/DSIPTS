uv venv --python 3.11
source .venv/bin/activate

uv pip install 'numpy<2.0.0' matplotlib pandas scikit-learn 'pytorch-lightning==1.9.4' hydra-core hydra-joblib-launcher hydra-optuna-sweeper 'beautifulsoup4==4.12.0' html5lib 'html-table-parser-python3==0.3.1' sphinx plotly scikit-learn numba einops sphinx_pdj_theme sphinx_mdinclude 'aim==3.20.1' transformers 
python setup_uv.py install --force