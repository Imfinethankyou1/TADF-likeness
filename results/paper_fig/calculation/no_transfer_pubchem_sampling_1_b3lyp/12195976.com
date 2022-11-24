%chk=./calculation/no_transfer_pubchem_sampling_1_b3lyp/12195976.chk
%nprocshared=7
%mem=6GB
#p b3lyp/6-31g(d) opt SCF=XQC 

Gaussian input prepared by ASE

0 1
N                 0.5491103342        0.0000000000        0.0000000000
N                -0.5491103342        0.0000000000        0.0000000000



