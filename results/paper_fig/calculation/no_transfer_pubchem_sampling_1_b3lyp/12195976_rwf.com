%chk=calculation/no_transfer_pubchem_sampling_1_b3lyp/12195976_rwf.chk
%rwf=calculation/no_transfer_pubchem_sampling_1_b3lyp/12195976_rwf.rwf
%nprocshared=14
%mem=3GB
#p b3lyp/6-31g(d) 6D 10F nosymm GFInput td=(50-50,nstate=2) SCF=XQC 

Gaussian input prepared by ASE

0 1
N                 0.5527487451        0.0000000000        0.0000000000
N                -0.5527487451        0.0000000000        0.0000000000



