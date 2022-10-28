import pandas as pd
import rdkit
from rdkit import Chem
import random
db = '23637518'

with open('solvent.txt') as f:
    lines=f.readlines()

solvent_list = []
for line in lines:
    solvent_list.append(line.strip().split()[0])
solvent_list.append('C1COCCO1')
solvent_list+=['CN(C)C=O', 'CS(C)=O','CC(C)=O','O','gas']
solvent_list += ['CC(C)(C)O', 'CNC=O', 'NC=O']
solvent_list +=['gas', 'C[Si](C)(C)C', 'C1CC[C@H]2CCCC[C@H]2C1', '[2H]C(Cl)(Cl)Cl', 'FC(Cl)(Cl)Cl', 'ClC(Cl)(Cl)Br', 'FC(F)(F)C(Cl)(Cl)Cl', 'Cl/C=C/Cl', 'Cl/C=C\\Cl', 'FC(F)(F)CCCl', 'CC(C)(Cl)Cl', 'FC(F)(F)C(F)(Cl)C(F)(F)Cl', 'ClC(Cl)=C(Cl)C(Cl)(Cl)Cl', 'ClCCCCCl', 'FC1(F)C(F)(F)C(F)(Cl)C1(F)Cl', 'ClC1CCCCC1', 'ClCCCCCCCCCCCl', 'Cc1ccc(C)c(C)c1C', 'C#Cc1ccccc1', 'c1ccc(C2CCCCC2)cc1', 'Fc1ccccc1F', 'Fc1cc(F)cc(F)c1', 'Fc1cc(F)c(F)c(F)c1F', 'Fc1c(F)c(F)c(F)c(F)c1F', 'Cc1cc(Br)ccc1Br', 'Cc1ccccn1', 'Cc1ccncc1', 'CC(C)(C)c1cccc(C(C)(C)C)n1', 'Cc1cc(C)nc(C)c1', 'Fc1ccccn1', 'Clc1ccccn1', 'Brc1cccnc1', 'Fc1cccc(F)n1', 'Fc1nc(F)c(F)c(F)c1F', 'N#Cc1ccccn1', 'COCCO', 'CCOCCO', 'OCCS', 'NCCO', 'N#CCCO', 'OCCCl', 'OCC(F)(F)F', 'OCC(Cl)(Cl)Cl', 'OCC(F)(F)C(F)F', 'C=CCO', 'C#CCO', 'OC(C(F)(F)F)C(F)(F)F', 'COCC(C)O', 'CC(O)CN', 'CCC(N)CO', 'CC(C)(C)O', 'CCC(C)CO', 'CC(C)C(C)O', 'CCC(O)(C(C)C)C(C)C', 'CCCCCCCCCCCO', 'CCCCCCCCCCCCO', 'OC1CCCC1', 'OCc1ccco1', 'OCC1CCCO1', 'CC(O)CO', 'OCCCO', 'CCC(O)CO', 'CC(O)CCO', 'OCCCCO', 'CC(O)C(C)O', 'OCCCCCO', 'OCCOCCO', 'OCCOCCOCCO', 'OCCOCCOCCOCCO', 'OCCSCCO', 'OCCN(CCO)CCO', 'Oc1ccccc1', 'Cc1ccccc1O', 'Cc1ccc(O)cc1', 'CCc1cccc(O)c1', 'CCc1ccc(O)cc1', 'CC(C)c1ccccc1O', 'CC(C)(C)c1ccccc1O', 'CC(C)(C)c1ccc(O)cc1', 'Cc1cccc(O)c1C', 'Cc1ccc(O)c(C)c1', 'Cc1ccc(O)cc1C', 'Cc1cc(C)cc(O)c1', 'Cc1ccc(C(C)C)c(O)c1', 'Cc1ccc(O)c(C(C)(C)C)c1', 'CC(C)(C)c1ccc(O)c(C(C)(C)C)c1', 'CC(C)(C)c1cccc(C(C)(C)C)c1O', 'Cc1cc(C)c(C)c(O)c1', 'Cc1ccc(C)c(O)c1C', 'Cc1cc(C(C)(C)C)c(O)c(C(C)(C)C)c1', 'Oc1ccc(F)cc1', 'Oc1ccc(Cl)cc1', 'Oc1ccc(Br)cc1', 'Oc1c(Cl)cccc1Cl', 'Cc1cc(O)ccc1Cl', 'COc1ccc(O)cc1', 'COc1cccc(OC)c1O', 'COc1cc(O)cc(OC)c1', 'CC(=O)Oc1cccc(O)c1', 'O=C(Oc1ccccc1)c1ccccc1O', 'Nc1ccccc1O', 'Nc1cccc(O)c1', 'O=[N+]([O-])c1ccccc1O', 'N#Cc1ccc(O)cc1', 'CCC(C)(C)OC', 'CCOCOCC', 'COCCOC', 'CCOCCOCCOCC', 'COCCOCCOCCOC', 'CC1CCC(C)O1', 'C1CCSC1', 'C1COCO1', 'CC1OCCO1', 'CC1CCCOC1', 'CC(=O)C1(C)COCOC1', 'c1ccc(COCc2ccccc2)cc1', 'c1ccc(Oc2ccccc2)cc1', 'CC(=O)C(Cl)(Cl)Cl', 'CC(=O)CC(C)C', 'CC(C)CC(=O)CC(C)C', 'O=C1CCCC1', 'CC(=O)OC(C)=O', 'COC(=O)C(Cl)(Cl)Cl', 'CCOC(=O)CCl', 'CCOC(=O)C(Cl)(Cl)Cl', 'C=COC(C)=O', 'COC(=O)OC', 'CCOC(=O)OCC', 'O=C1OCCO1', 'CC1COC(=O)O1', 'CCOC(=O)CC(C)=O', 'CCOC(=O)[C@H](C)O', 'NC=O', 'CNC=O', 'CN(C)C=S', 'CN(C=O)c1ccccc1', 'CNC(C)=O', 'CCN(CC)C(C)=O', 'CN1CCCC1=S', 'CCN1CCCC1=O', 'O=C1NCCC1C1CCCCC1', 'O=C1CCCCCN1CO', 'CN(C)C#N', 'CCN(C#N)CC', 'CC(C)N(C#N)C(C)C', 'N#CN1CCCC1', 'N#CN1CCCCC1', 'N#CN1CCOCC1', 'CN(C)C(=N)N(C)C', 'CN1CCN(C)C1=O', 'CN1CCCN(C)C1=O', 'COCCC#N', 'CCCC#N', 'N#CCc1ccccc1', 'O=[N+]([O-])C1CCCCC1', 'CC(C)(C)N', 'NCCN', 'CCNCC', 'COP(=O)(OC)OC', 'CCCOP(=O)(OCCC)OCCC', 'CN(C)P(=O)(N(C)C)N(C)C', 'CCN(CC)P(=O)(N(CC)CC)N(CC)CC', 'CCP(C)(=O)N(C)C', 'CSCS(C)=O', 'O=S1(=O)CCCC1', 'CC1CCS(=O)(=O)C1', 'CCN(CC)S(=O)(=O)N(CC)CC']
solvent_list +=['CCCCCCCCCCCCCCCCCl', '[2H]O[2H]', 'CCCCCc1ccc(-c2ccc(C#N)cc2)cc1', 'CCCCCCc1ccc(-c2ccc(C#N)cc2)cc1', 'CCCCc1ccc(/N=C/c2ccc(OC)cc2)cc1', 'CC(C)CCCC(C)CCCC(C)CCCCC(C)CCCC(C)CCCC(C)C', 'O=C(O)C(F)(F)F', 'CCCCCCCCCCCCC', 'CCCCCCCCCCCCCCC']
solvent_list += ['CCCCCOCCCCC','[2H]OC', '[2H]OCC','C1CCC2CCCCC2C1', 'C1CCC2CCCCC2C1', '[2H]OC([2H])([2H])C(F)(F)F']
#['[2H]OC', '[2H]OCC']

solvent_list = set(solvent_list)
csv = pd.read_csv(db)
with open('host_list.txt') as f:
    lines=f.readlines()
host_list = []
for line in lines:
    smiles=line.strip().split()[-1]
    host_list.append(smiles)


count = 0
smiles_list = []
sol_list = []
for i in range(len(csv['Solvent'])):
    smiles_list.append(csv['Chromophore'][i])
    sol_list.append(csv['Solvent'][i])    

smiles_list_ = [ Chem.MolToSmiles(Chem.MolFromSmiles(smiles))   for smiles in smiles_list]
smiels_list = smiles_list_

smiles2sol = {}
for smiles in smiles_list:
    smiles2sol[smiles] = []

for smiles, sol in zip(smiles_list, sol_list):
    if smiles in smiles2sol:
        smiles2sol[smiles] += [sol]
    else:
        smiles2sol[smiles] = [sol]

smiles_list = []



negative_smiles_list = []
for smiles in smiles2sol.keys():
    sol_list = smiles2sol[smiles]
    sol_list = list(set(smiles2sol[smiles]))
    sol_list_ = sol_list[:]
    for sol in sol_list_:
        if sol == smiles:
            sol_list.remove(sol)
        else:
            if sol in solvent_list:
                sol_list.remove(sol)
    if len(sol_list) != 0:
        smiles_list.append(smiles)
    else:
        negative_smiles_list.append(smiles)
smiles_list = list(set(smiles_list))

count = 0
f = open('TADF_dopant.txt','w')
for i, smiles in enumerate(smiles_list):
    #if not smiles in ref_smiles_list:
    f.write(f"{i} {smiles}\n")
f.close()
import sys
sys.exit()
g = open('TADF_dopant_test_negative.txt','w')
for i, smiles in enumerate(negative_smiles_list):
    if not smiles in ref_smiles_list:
        if count < 1000:
            g.write(f"{i} {smiles}\n")
        count+=1
g.close()
print(count)
