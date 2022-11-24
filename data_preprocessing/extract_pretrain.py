import rdkit
from rdkit import Chem

#with open('../data/total_train_data.txt') as f:
with open('total_TADF.txt') as f:
    lines = f.readlines()

smiles_list = [Chem.MolToSmiles(Chem.MolFromSmiles(line.strip().split()[1])) for line in lines]

with open('vis_chromophore.txt') as f:
    lines = f.readlines()

#new_lines = []
with open('vis_chromophore_pretrain.txt','w') as f:
    for line in lines:
        smiles = line.strip().split()[1]
        if not smiles in smiles_list:
            f.write(line)


