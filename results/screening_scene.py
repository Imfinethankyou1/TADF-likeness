import pickle
import subprocess


with open('../data_preprocessing/smiles2ref.pickle', 'rb') as f:
    data = pickle.load(f)

#with open('upper_chromophore_list.txt') as f:
#    lines = f.readlines()

with open('EF-100.txt') as f:
    lines = f.readlines()[1:]


s2line= {}
for line in lines:
    smiles = line.strip().split()[0]
    s2line[smiles] = line


words = ['thermally', 'Thermally','delayed']

count = 0
f = open('scene_result.txt','w')
for smiles in s2line.keys():
    if smiles in data.keys():
        dois=data[smiles]
        #output =subprocess.check_output(['doi2bib','10.1002/adom.201900476'])
        #print(doi)
        TADF = False
        TADF_doi = ''
        for doi in dois:
            output =subprocess.check_output(['doi2bib',doi])
            for word in words:
                if word in str(output):
                    TADF = True
                    TADF_doi = doi

        if True and  s2line[smiles].strip().split()[-1] != 'TADF':
            count +=1
            print(s2line[smiles].strip(), TADF_doi)
            #f.write(f'{s2line[smiles].strip()} {TADF_doi}\n')
            f.write(f'{s2line[smiles].strip()} {dois}\n')
print(count, len(lines))
