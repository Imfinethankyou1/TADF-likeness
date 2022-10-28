# TADF-likeness


## How to data processing
-----------------------------

1. Make chromophore database
```yaml 
    cd data_preprocessing
    python extract_fluorescence.py
    python data_preprocessing/extract_pretrain.py
```
2. Random sampling from Pubchem
-> this routine needs to a lot of time-cost
-> Thus, we prepared this data (data/random_2_log5.txt)

3. Generate molecular descriptors using prepared database
```yaml
    cd data
    python make_input.py
    python make_database.py #(25m 53.0)
     
