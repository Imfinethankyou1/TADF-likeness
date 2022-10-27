import pandas as pd
import random
import csv
import sys
from rdkit import Chem
from rdkit.Chem import AllChem
import os
from ase.io import read
from multiprocessing import Pool
random.seed(0)


def smiles_to_xyz_with_scaffold(input_dict):
    '''
        Extract scaffold from SMILES, and make UFF xyz 
        return INDEX, SMILES, scaffold SMILES
    '''
    smiles = input_dict['smiles']
    filename = input_dict['filename']
    ind = input_dict['ind']
    rest = input_dict['rest']
    except_val = 0
    except_list = ['F', 'Cl', 'Br','I','U', 'Te', 'Se', 'Tl', 'Tc','+','.','P=O', 'S=O','Pb','Mn', '-','A','G','Hg']
    for element in except_list:
        if element in smiles:
            except_val = 1
    data_line = '' 
    if except_val == 0:
        #try:
            print(smiles)
            mol = Chem.MolFromSmiles(smiles)
            atomnum = mol.GetNumAtoms()
            if atomnum <= 60 and atomnum > 6:
                mol2 = Chem.AddHs(mol)
                AllChem.EmbedMolecule(mol2)
                AllChem.UFFOptimizeMolecule(mol2)
                Chem.rdmolfiles.MolToXYZFile(mol2,filename)
                data_line = str(ind)+" "+smiles+' '.join(input_dict[rest])+' \n'
        #except:
        #    print('error smiles : ',smiles)
    return  data_line

def canonical_and_rm_duple(smiles_list):
    smiles_list_ = [ Chem.MolToSmiles(Chem.MolFromSmiles(smiles))   for smiles in smiles_list]
    smiles_list = list(set(smiles_list_))
    return smiles_list

if __name__ == '__main__':
    os.system('wget https://figshare.com/ndownloader/files/23637518')
    
    csv = pd.read_csv('23637518')

    min_val = 500
    smiles_list = []
    feature_list = []
    vis_smiles_list = []
    for i in range(len(csv['Solvent'])):
        sol = csv['Solvent'][i]
        #if sol == 'Cc1ccccc1':
        if True:
            if str(csv['Emission max (nm)'][i]) != 'nan':
                color =  float(str(csv['Emission max (nm)'][i]))
                eV = 1240/color
                smiles_list.append(csv['Chromophore'][i])
                if 1.6 < eV < 3.1: # make visible_range_fluorescence
                    vis_smiles_list.append(csv['Chromophore'][i])
    smiles_list = canonical_and_rm_duple(smiles_list)
    vis_smiles_list = canonical_and_rm_duple(vis_smiles_list)
    with open('total_chromophore.txt','w') as f:
        for i, smiles in enumerate(smiles_list):
            f.write(f"{i} {smiles}\n")

    with open('vis_chromophore.txt','w') as f:
        for i, smiles in enumerate(vis_smiles_list):
            f.write(f"{i} {smiles}\n")

    count = len(smiles_list)
    print('Total toluene solvent data : ', count)
