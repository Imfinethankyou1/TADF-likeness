%chk=calculation/no_transfer_pubchem_sampling_1_b3lyp/108151406_rwf.chk
%rwf=calculation/no_transfer_pubchem_sampling_1_b3lyp/108151406_rwf.rwf
%nprocshared=14
%mem=3GB
#p b3lyp/6-31g(d) 6D 10F nosymm GFInput td=(50-50,nstate=2) SCF=XQC 

Gaussian input prepared by ASE

0 1
C                -4.7932250671       -1.3356726745       -0.8386039013
C                -3.4409517704       -1.3700141770       -0.1194881777
C                -2.6924199487       -0.0311483777       -0.1770936911
C                -1.3394638203       -0.0707723545        0.5466108065
C                -0.4935568313        1.2112458999        0.3977014510
C                -1.1275482970        2.4433549799        1.0500182615
N                 0.8476828373        1.0290879856        0.9609215058
C                 1.8682883065        0.4042060113        0.3163664321
N                 1.7407152172        0.0502627448       -0.9744029088
C                 2.8077836286       -0.5249363615       -1.5393071709
N                 3.9853605216       -0.8032792760       -0.9860579303
C                 4.1281043586       -0.4607947513        0.3177520600
C                 5.4348531285       -0.7544907406        1.0026825106
C                 3.0836173941        0.1428205113        0.9821665003
F                 3.1918217271        0.4971599054        2.2926284155
H                -4.6714566939       -1.0862299386       -1.8998464680
H                -5.3012308924       -2.3050118605       -0.7808794594
H                -5.4591897837       -0.5840326116       -0.3968487388
H                -2.8104633704       -2.1547428066       -0.5602662808
H                -3.5918037915       -1.6554321619        0.9314113056
H                -2.5319902329        0.2468949646       -1.2294583322
H                -3.3324143994        0.7515248066        0.2533604809
H                -0.7477558706       -0.9044310399        0.1485995096
H                -1.4944309872       -0.2752841926        1.6168367829
H                -0.3470836505        1.4016585267       -0.6701607339
H                -2.1063173869        2.6587506218        0.6121990837
H                -0.4915284562        3.3240563676        0.9145635500
H                -1.2741000065        2.2879421256        2.1277243202
H                 0.9640424156        1.1572028648        1.9568472496
H                 2.6920853913       -0.8027373685       -2.5856616593
H                 5.2920475176       -1.4253849789        1.8579380556
H                 6.1149772225       -1.2244124557        0.2898985889
H                 5.8955505904        0.1626418122        1.3878545817


