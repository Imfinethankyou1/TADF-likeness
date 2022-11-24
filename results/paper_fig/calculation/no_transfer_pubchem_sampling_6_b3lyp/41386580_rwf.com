%chk=calculation/no_transfer_pubchem_sampling_6_b3lyp/41386580_rwf.chk
%rwf=calculation/no_transfer_pubchem_sampling_6_b3lyp/41386580_rwf.rwf
%nprocshared=14
%mem=3GB
#p b3lyp/6-31g(d) 6D 10F nosymm GFInput td=(50-50,nstate=2) SCF=XQC 

Gaussian input prepared by ASE

0 1
C                -4.9243889603        1.3701062200        2.7407715985
C                -4.1267796454        0.4569362508        1.8393647969
C                -4.0625235615        0.6668329915        0.4599751455
C                -3.3100914918       -0.1681517630       -0.3715676306
C                -2.5902533618       -1.2467401346        0.1527341818
C                -1.7488430760       -2.1999099427       -0.6942198278
C                -1.9537956793       -2.0733120406       -2.2112400329
C                -1.3460981325       -0.8189701532       -2.8379460493
O                -1.8603629201       -0.3310473767       -3.8394067978
C                -0.1304496182       -0.2656759101       -2.2040881780
C                 0.5683726106        0.8697828806       -2.7113985098
C                 0.1470931993        1.6228960623       -3.9430505987
N                 1.6578916346        1.3281487045       -2.1043223874
C                 2.0727005624        0.6830476862       -0.9880278494
N                 3.2008942638        1.2287280203       -0.4424277990
C                 3.9733021765        0.9037139894        0.6866998936
C                 5.0914734624        1.7200515707        0.9346883477
C                 5.9173279300        1.4774613700        2.0270982993
C                 5.6459028344        0.4169220720        2.8942411901
C                 4.5366439896       -0.3921612849        2.6475054545
C                 3.6978000728       -0.1632314697        1.5564983029
N                 1.4865829189       -0.3806052291       -0.4200955927
C                 0.3934035391       -0.8461844246       -1.0318333952
C                -0.2369043154       -2.0657220864       -0.4033597865
C                -2.6654991427       -1.4635783213        1.5385411216
C                -3.4145923170       -0.6320763333        2.3641673807
H                -5.5472417541        0.8003927283        3.4405095145
H                -4.2668176336        2.0099326362        3.3439133902
H                -5.5824593565        2.0270097388        2.1631064959
H                -4.6114729098        1.4960565421        0.0191418549
H                -3.3043692852        0.0346796316       -1.4377774295
H                -2.0372571847       -3.2190239467       -0.4041388186
H                -1.4674942789       -2.9325204583       -2.6979001649
H                -3.0086590168       -2.1087533595       -2.4976601998
H                 0.8570301527        2.4337436512       -4.1193241792
H                 0.1038474607        0.9621000519       -4.8138221502
H                -0.8638849300        2.0255455134       -3.8293843602
H                 3.5198962299        2.0259797723       -0.9790172692
H                 5.3096334029        2.5485886444        0.2636054586
H                 6.7756698891        2.1213242590        2.1990796387
H                 6.2894610761        0.2257273341        3.7480050186
H                 4.3122571361       -1.2213355939        3.3136613305
H                 2.8393321239       -0.7920053881        1.3695163381
H                -0.0301374059       -2.0430567645        0.6701377837
H                 0.2815968821       -2.9517850668       -0.8010044373
H                -2.1339696408       -2.3062464557        1.9772647211
H                -3.4537679297       -0.8336148181        3.4327861859


