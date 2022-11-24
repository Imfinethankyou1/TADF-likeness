%chk=calculation/no_transfer_pubchem_sampling_0_b3lyp/59258646_rwf.chk
%rwf=calculation/no_transfer_pubchem_sampling_0_b3lyp/59258646_rwf.rwf
%nprocshared=14
%mem=3GB
#p b3lyp/6-31g(d) 6D 10F nosymm GFInput td=(50-50,nstate=2) SCF=XQC 

Gaussian input prepared by ASE

0 1
O                -2.4581220430       -3.3394812084       -0.9479619106
C                -2.5202522711       -2.2056920511       -0.4854225581
N                -3.5124616433       -1.7281737378        0.3262380081
C                -4.6482740123       -2.3932413948        0.8044832566
C                -4.9629094318       -3.7285755740        0.5144262135
C                -6.1293951718       -4.2565290494        1.0633268174
C                -6.9363107995       -3.4561796629        1.8712403120
N                -6.6442867038       -2.1791241229        2.1542049726
C                -5.5301779156       -1.6736481182        1.6301783079
C                -1.4572297750       -1.1671669420       -0.7810775829
C                -0.4122507602       -1.5022345325       -1.6177085426
C                 0.5328775002       -0.4810503055       -1.8395000136
O                 1.5407457158       -0.7759159774       -2.6735485539
C                 2.6501271717        0.1261449210       -2.8307517550
C                 3.8953677852       -0.4659590468       -2.1712643029
N                 3.7827433511       -0.5619518750       -0.7254978557
C                 3.8774422708        0.7407437501       -0.0538753045
C                 4.2173813935        0.5552085652        1.4203592748
O                 5.4924179524       -0.0511268189        1.6162106022
C                 5.8558612014       -0.9873530006        0.6010509024
C                 4.6294825500       -1.5856442229       -0.0879573028
N                 0.4356477551        0.7211910717       -1.2882268759
C                -0.6364710887        0.9426126655       -0.4912914973
N                -0.7501148441        2.1811079243        0.0711732983
C                 0.1544706431        3.2766101736       -0.2765709327
C                -0.6045715092        4.6000110273       -0.2196337075
O                -1.1821019524        4.8678039682        1.0559379610
C                -1.4561793021        3.7092207216        1.8359203062
C                -1.8396036445        2.4925036599        0.9982222206
N                -1.5977156639        0.0441492252       -0.2098997783
H                -3.3797892685       -0.7574171926        0.5982290654
H                -4.3084013257       -4.3133489065       -0.1174472658
H                -6.4082792943       -5.2869449152        0.8624027084
H                -7.8514490409       -3.8489570732        2.3098549088
H                -5.3102777821       -0.6321384546        1.8719328534
H                -0.3259242339       -2.4825601669       -2.0646475225
H                 2.3904167246        1.1063872332       -2.4304290668
H                 2.8157278410        0.2133571457       -3.9093519666
H                 4.7616849814        0.1479876140       -2.4980188065
H                 4.0389407848       -1.4723832639       -2.5782163739
H                 4.6561165720        1.3799754226       -0.5150473309
H                 2.9228578462        1.2740273877       -0.1432701482
H                 4.2562655007        1.5275605834        1.9223239777
H                 3.4289268720       -0.0412662107        1.9059078044
H                 6.5030499967       -0.4966802719       -0.1441008603
H                 6.4468297170       -1.7723207187        1.0877226297
H                 4.9485531621       -2.3150805753       -0.8397061122
H                 4.0400525633       -2.1377065669        0.6572194466
H                 0.5385596093        3.1133734267       -1.2853895813
H                 1.0195359813        3.3052183341        0.4026296176
H                -1.3909345328        4.5968443124       -0.9903071876
H                 0.0786395102        5.4272273065       -0.4415971457
H                -2.2826249773        3.9704590842        2.5044676314
H                -0.5856703943        3.4581531363        2.4625321876
H                -2.0123332789        1.6351231352        1.6546475758
H                -2.7766062922        2.6828481620        0.4548779824


