# TADF-likeness


## How to install environment
-----------------------------
    conda create -n TL
    source activate TL
    conda install -c conda-forge mamba
    mamba install xtensor-r -c conda-forge
    mamba install pytorch==1.12.1 torchvision==0.13.1 torchaudio==0.12.1 cudatoolkit=10.2 -c pytorchhttps://download.pytorch.org/whl/torch_stable.html
    pip install rdkit-pypi
    pip install git+https://github.com/bp-kelley/descriptastorus
    pip install seaborn
    pip install sklearn


## How to data processing
-----------------------------

1. Make chromophore database
```yaml 
    cd data_preprocessing
    python extract_fluorescence.py
    python extract_pretrain.py # for pretraining
    python utils.py # Making training dataset
```
2. Random sampling from Pubchem
-> this routine needs to a lot of time-cost
-> Thus, we prepared this data (data/random_2_log5.txt)

3. Generate molecular descriptors using prepared database
```yaml
    cd data
    python make_input.py
    python make_database.py #(25m 53.0)
```

     
## How to execute 
-----------------
```yaml
   source train_vis_chromophore.sh
   source train_TADF.sh
```


## How to see the results in our paper
-----------------
There are DFT calculation results, TADF-likenss scores of molecules in various database, and codes for showing results in results directory
You can download real world scenario data at https://www.dropbox.com/sh/9zoq6kc42vutlwc/AAANNOvIAGwMLoZKdbjJqVCha?dl=0
