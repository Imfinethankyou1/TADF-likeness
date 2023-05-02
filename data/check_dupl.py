import rdkit
from rdkit import Chem


with open('total_train_data.txt') as f:
#with open('TADF-2022.txt') as f:
    lines = f.readlines()

with open('vis_chromophore_pretrain.txt') as f:
#with open('unseen-TADF.txt') as f:
    lines += f.readlines()

smiles_list = []
for line in lines:
    smiles = line.strip().split()[1]
    smiles = Chem.MolToSmiles(Chem.MolFromSmiles(smiles))
    smiles_list.append(smiles)
print(len(smiles_list))

#with open('sample/TADF-likeness-property4.txt') as f:
with open('TADF-2022.txt') as f:
    lines = f.readlines()

ch_smiles_list =[]
for line in lines:
    smiles = line.strip().split()[1]
    smiles = Chem.MolToSmiles(Chem.MolFromSmiles(smiles))
    if not smiles in smiles_list:
        ch_smiles_list.append(smiles)

ch_smiles_list = list(set(ch_smiles_list))

print(len(ch_smiles_list), len(lines))

#with open('total_chromophore_rm_duple.txt','w') as f:
#    for i, smiles in enumerate(ch_smiles_list):
#        f.write(f'{i} {smiles}\n')

#print(len(smiles_list))
