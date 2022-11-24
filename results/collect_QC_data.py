import glob
import os
import matplotlib.pylab as plt
import seaborn as sns
from cclib.parser import Gaussian as G
import math
import numpy as np

def parse_homo_lumo(mo_e, homo):
    return {"HOMO": mo_e[homo], "LUMO": mo_e[homo+1]}

def read_data(fn,ind2smiles):
    count = 0
    s1_list = []
    t1_list = []
    f1_list = []
    homo_list = []
    lumo_list = []
    normal = 0
    with open(fn) as f:
        lines= f.readlines()
    prefix = fn.replace('sampling/','calculation/')
    prefix = prefix.replace('.txt','_b3lyp/')
    filenames = []
    cid2score = {}
    for line in lines:
        cid=line.split()[0]
        filename = prefix+cid+'_rwf.log'
        #print(filename)
        if os.path.exists(filename):
            filenames.append(filename)
            cid2score[cid]=line.strip().split()[-1]
    outputs = []
    #print(filenames)
    for filename in filenames:
        ind = filename.split('/')[-1].split('_rwf')[0]
        if os.path.isfile(filename):
            with open(filename) as f:
                lines = f.readlines()
            if 'Normal' in lines[-1]:
                log_parser = G(filename)
                parsed_data = log_parser.parse().getattributes()
                normal +=1
                a=parse_homo_lumo(parsed_data["moenergies"][0], parsed_data["homos"][0])
                homo = a['HOMO']
                lumo = a['LUMO']
                s_list = []
                t_list = []
                f_list = []
                for line in lines:
                    if 'Singlet' in line:
                        f= float(line.split()[8].split('=')[-1])
                        f_list.append(f)
                        s = float(line.split()[4])
                        s_list.append(s)
                    if 'Triplet' in line:
                        t = float(line.split()[4])
                        t_list.append(t)
                f = math.log10(f_list[0] + 1e-5 )
                score = cid2score[ind]
                smiles = ind2smiles[ind]
                line = f"{ind} {smiles} {homo} {lumo} {s_list[0]} {t_list[0]} {f} {score}\n"
                outputs.append(line)

    return outputs

ind2smiles ={}
fns = [f'sampling/no_transfer_pubchem_sampling_{i}.txt' for i in range(5,10)]
print(fns)
for fn in fns:
    with open(fn) as f:
        lines = f.readlines()
    for line in lines:
        ind, smiles =line.strip().split()[:2]
        ind2smiles[ind] = smiles
lines = []
for fn in fns:
    lines+=read_data(fn,ind2smiles)

with open('total_QC_data.txt','w') as f:
    for line in lines:
        f.write(line)
