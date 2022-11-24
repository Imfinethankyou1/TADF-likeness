%chk=calculation/no_transfer_pubchem_sampling_3_b3lyp/64594203_rwf.chk
%rwf=calculation/no_transfer_pubchem_sampling_3_b3lyp/64594203_rwf.rwf
%nprocshared=14
%mem=3GB
#p b3lyp/6-31g(d) 6D 10F nosymm GFInput td=(50-50,nstate=2) SCF=XQC 

Gaussian input prepared by ASE

0 1
N                -5.5764439442       -1.5958089763       -0.2039192044
C                -4.5893000392       -1.0181354315       -0.7935077644
N                -4.5103801453       -0.5915830418       -2.0970114319
C                -3.3599597992       -0.7161062506        0.0153283080
C                -3.3347595783       -0.9560412781        1.3900317709
C                -2.1578721196       -0.6655065226        2.0765977053
C                -1.0594011673       -0.1501986165        1.3922962128
C                -1.1998130898        0.0543450797        0.0135033395
O                -0.1912507514        0.5281852731       -0.7890081327
C                 0.9594150223        1.0637524553       -0.2249195255
C                 0.9331582842        2.3104340000        0.3580137585
C                 2.1276607397        2.8743965150        0.8641919671
C                 3.3182017029        2.1898342792        0.7706992324
C                 3.3714825384        0.9066701729        0.1631887909
C                 4.5840706031        0.1748715562        0.0462410314
C                 4.6066124234       -1.0668360124       -0.5469196151
C                 3.4142928072       -1.6376245116       -1.0557382906
C                 2.2196752948       -0.9598312190       -0.9616126904
C                 2.1673924870        0.3208700577       -0.3502624244
N                -2.3098352021       -0.2201725744       -0.6589436049
H                -6.3567999948       -1.7415326175       -0.8518625666
H                -3.5862649383       -0.3362320211       -2.4241310733
H                -5.1512710564       -0.9823541444       -2.7719796266
H                -4.2153704881       -1.3580422339        1.8746392581
H                -2.0893978378       -0.8418923440        3.1466953338
H                -0.1314264154        0.0803863560        1.9007479681
H                -0.0057616103        2.8519479816        0.4178145677
H                 2.0959013211        3.8583245772        1.3233627152
H                 4.2361695246        2.6250097931        1.1574039131
H                 5.4983646848        0.6169663972        0.4348382915
H                 5.5418994936       -1.6138195449       -0.6292384743
H                 3.4457722007       -2.6169735863       -1.5252593070
H                 1.3052400497       -1.3873045672       -1.3584734323


