import subprocess
import glob
import matplotlib.pyplot as plt
import seaborn as sns
from rdkit import Chem
from rdkit.Chem import Draw
import os
import xlsxwriter
#from pil import image
from ase.io import read
import rdkit
from rdkit import Chem
from rdkit.Chem.Draw import rdMolDraw2D

def drawMyMol(mol, fname, myfontsize):
    d = rdMolDraw2D.MolDraw2DCairo(1000, 1000)
    d.SetFontSize(myfontsize)
    #print(d.fontsize())
    d.DrawMolecules([mol])
    d.FinishDrawing()
    d.WriteDrawingText(fname)

with open('../make_supporting_data/head_line.txt') as f:
    lines = f.readlines()

with open('table_examples.txt') as f:
    examples = f.readlines()

with open('table1.txt','w') as f:
    for line in lines:
        f.write(line)

    for example in examples[1:]:
        cid, smiles, likeness, homo, lumo, s1, est= example.strip().split()
        
        mol = Chem.MolFromSmiles(smiles)
        fig_name = 'examples/'+cid+'.png'
        drawMyMol(mol, fig_name, 6)
         
        new_line = '\includegraphics[width=10em]{'+f'{fig_name}'+'} & ' f'{homo} & {lumo} & {s1} & {est} & {likeness} \\\\ [0.5ex] \n'
        new_line += '\hline \n'
        f.write(new_line)

