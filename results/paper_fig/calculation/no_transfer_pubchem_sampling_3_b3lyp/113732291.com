%chk=./calculation/no_transfer_pubchem_sampling_3_b3lyp/113732291.chk
%nprocshared=7
%mem=6GB
#p b3lyp/6-31g(d) opt SCF=XQC 

Gaussian input prepared by ASE

0 1
C                -0.8938081113       -1.5880069110        2.1176594702
C                -1.3127802215       -0.6577054579        1.0253770115
C                -2.6044928148       -0.7015339088        0.5192651967
C                -2.9982560400        0.1609552713       -0.4938603343
C                -4.3932681782        0.0706948933       -1.0429403111
F                -5.2878682263       -0.3521278739       -0.1372431317
F                -4.4700342116       -0.7966632423       -2.0747844156
F                -4.8478830817        1.2430201108       -1.5108574994
C                -2.1061449811        1.0773113932       -1.0325600750
C                -0.8134213328        1.1165261763       -0.5452356766
C                -0.4108187430        0.2596014699        0.4739117144
C                 0.9847769171        0.3857971352        1.0005078032
O                 1.2115184296        0.5653142072        2.1777045796
C                 2.0648893389        0.3115531341       -0.0110445329
C                 3.3152497982        0.8389141578        0.3115670664
C                 4.3530227989        0.7811104064       -0.5978458242
C                 4.1574273800        0.1811606533       -1.8321428542
C                 2.9232029030       -0.3627751782       -2.1530150752
C                 1.8775068050       -0.2974444618       -1.2513314898
H                -0.5337575747       -1.0199071489        2.9729098985
H                -0.0760233809       -2.2251248847        1.7821022581
H                -1.7234855915       -2.2155718136        2.4305843982
H                -3.3154958225       -1.4059895493        0.9245891907
H                -2.4200982681        1.7468784320       -1.8177979616
H                -0.1096637664        1.8282429322       -0.9505754883
H                 3.4474674808        1.2967061732        1.2806264460
H                 5.3158500803        1.1996798665       -0.3458175093
H                 4.9684601508        0.1324853591       -2.5434848971
H                 2.7766925507       -0.8398252050       -3.1106378229
H                 0.9212367129       -0.7332761365       -1.5013141345


