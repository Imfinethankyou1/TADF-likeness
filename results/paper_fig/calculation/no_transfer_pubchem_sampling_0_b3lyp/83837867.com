%chk=./calculation/no_transfer_pubchem_sampling_0_b3lyp/83837867.chk
%nprocshared=7
%mem=6GB
#p b3lyp/6-31g(d) opt SCF=XQC 

Gaussian input prepared by ASE

0 1
C                -1.9528758548        2.7147829926        0.5832090047
C                -1.6310409738        1.2658642318        0.4070773055
N                -0.3356779855        0.9653173123        0.2909794173
C                 0.0423896846       -0.2997064329        0.1327934659
N                 1.3563508421       -0.5832971255        0.0524113999
C                 2.3847990662        0.4282238262       -0.0203726663
C                 3.5780106219        0.0022353843        0.8293900107
C                 2.7826159156        0.6815032616       -1.4960245624
N                 3.6188350859        1.8390453396       -1.7249036014
C                -0.9454925783       -1.3089575451        0.0870217152
C                -1.0816357803       -2.7068240908       -0.0564494368
N                -2.3453111786       -3.0549819409       -0.0166702228
N                -3.0720927651       -1.9372999248        0.1494088702
C                -4.5019140027       -1.9690893534        0.2341790200
C                -2.2778035085       -0.8515875571        0.2187250651
N                -2.6459798951        0.4231610260        0.3788961683
H                -3.0281487457        2.8434501360        0.6532883953
H                -1.4753660794        3.0887310643        1.4862999986
H                -1.5647201781        3.2813820214       -0.2602700220
H                 1.6180099842       -1.5316255194       -0.1659702944
H                 1.9325728896        1.3404454283        0.3833882812
H                 4.0251274245       -0.9059977941        0.4280703714
H                 3.2619247033       -0.1868263912        1.8522580776
H                 4.3352065324        0.7821065076        0.8455186839
H                 1.8653225473        0.8171780450       -2.0761368199
H                 3.3088584805       -0.1953202670       -1.8863659663
H                 3.1549892734        2.6777167466       -1.3928957116
H                 4.5001242662        1.7505935228       -1.2314925120
H                -0.3224846704       -3.4537212738       -0.1837191911
H                -4.9244200217       -2.3900062209       -0.6796546728
H                -4.8444103616       -0.9448396025        0.3667548365
H                -4.8157617380       -2.5816558070        1.0809615926


