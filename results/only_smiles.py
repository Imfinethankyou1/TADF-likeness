import pickle
from rdkit import Chem
import sys
import requests
import time
from multiprocessing import Pool
from bs4 import BeautifulSoup, Comment
import pandas as pd
import nltk
from urllib.request import urlopen
from bs4 import BeautifulSoup
from requests_html import HTMLSession
from requests import get  

#with open('scene_result.txt') as f:
with open('upper_chromophore_list.txt') as f:
#with open('../data/vis_chromophore.txt') as f:
    lines = f.readlines()

smiles_list = []
for line in lines:
    smiles_list += [line.strip().split()[1]]

with open('../data_preprocessing/smiles2sol.pickle', 'rb') as f:
    data = pickle.load(f)

with open('../data_preprocessing/smiles2ref.pickle', 'rb') as f:
    ref = pickle.load(f)

film_elements = ['film', 'Film', 'Neat', 'neat']
TADF_elements = ['TADF', 'Thermally', 'delayed']
count = 0

doi2mols = {}
for smiles in smiles_list:
    for sol in data[smiles]:
        if not 'gas' in sol:
            sol = Chem.MolToSmiles(Chem.MolFromSmiles(sol))
            if sol == smiles:
                #print(smiles, sol, ref[smiles])
                film = False
                TADF = False
                dois = list(set(ref[smiles]))
                for doi in dois:
                    url = 'https://doi.org/'+doi.split()[-1]
                    if not url in doi2mols.keys():
                        doi2mols[url] = [smiles]
                    else:
                        doi2mols[url] += [smiles]
                    if False:
                        try:
                            response = get(url)               
                            paper= str(response.content)
                            paper = paper.lower()
                            for element in film_elements:
                                if element in paper:
                                    film = True
                            for element in TADF_elements:
                                if element in paper:
                                    TADF = True
                            print(url, film, TADF)
                        except:
                            a= 1
                if TADF and film:
                    print(smiles, sol, ref[smiles])
                    count +=1

#doi_list = list(set(doi_list))
print('count : ',count)       
print(len(doi2mols.keys()))

for key in doi2mols.keys():
    #print(key,doi2mols[key])

    print(key, end = ' ')
    for smiles in doi2mols[key]:
        print(smiles, end=' ')
    print(len(doi2mols[key]))
