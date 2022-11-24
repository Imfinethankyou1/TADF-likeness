%chk=calculation/no_transfer_pubchem_sampling_1_b3lyp/154086091_rwf.chk
%rwf=calculation/no_transfer_pubchem_sampling_1_b3lyp/154086091_rwf.rwf
%nprocshared=14
%mem=3GB
#p b3lyp/6-31g(d) 6D 10F nosymm GFInput td=(50-50,nstate=2) SCF=XQC 

Gaussian input prepared by ASE

0 1
C                -1.6866967143       -3.4397932840        1.6242742012
C                -1.8115925227       -2.0777030210        0.9797395938
C                -0.6443925072       -1.3600575655        0.6423244968
C                -0.7048182598       -0.1056202479        0.0487430782
S                 0.7257383473        0.8695990546       -0.4100486107
N                 2.0106775658       -0.1836615559        0.0905384683
C                 2.8502284513        0.3421716171        1.1850081605
C                 3.8970773477        1.3362145104        0.6788140333
O                 4.6940271394        0.7551240407       -0.3487465843
C                 3.9009147027        0.3281399969       -1.4520751974
C                 2.8541782162       -0.6951437697       -1.0076899972
C                -1.9685870720        0.4559269229       -0.2191920413
N                -2.1744563952        1.6945429598       -0.8055446187
C                -3.4339343935        1.9576003906       -0.9322536123
C                -3.9808834076        3.2166369841       -1.5291693779
S                -4.5343332794        0.6851037381       -0.3320206106
C                -3.1310551606       -0.2666292230        0.1205832504
C                -3.0674445864       -1.5280715829        0.7174934499
H                -2.6685922469       -3.8771325370        1.8300202228
H                -1.1379568730       -4.1373424071        0.9793167403
H                -1.1397849998       -3.3831130290        2.5736564917
H                 0.3328346500       -1.7866525434        0.8459145390
H                 3.3652318273       -0.5167300633        1.6375814910
H                 2.2057559881        0.7946341172        1.9433755847
H                 3.3992915183        2.2475103625        0.3050196838
H                 4.5879787128        1.6236823197        1.4781208660
H                 3.4032990380        1.1944280996       -1.9210063616
H                 4.5945656904       -0.1065331568       -2.1792487054
H                 2.2125706982       -0.9953783835       -1.8403885350
H                 3.3693138723       -1.5890828926       -0.6291783314
H                -4.6043376031        3.7612587231       -0.8104698325
H                -4.5985568750        3.0053645162       -2.4099160728
H                -3.1454546486        3.8540584978       -1.8274616624
H                -3.9708082205       -2.0733525890        0.9738867995


