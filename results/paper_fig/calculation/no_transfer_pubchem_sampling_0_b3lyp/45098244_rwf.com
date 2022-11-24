%chk=calculation/no_transfer_pubchem_sampling_0_b3lyp/45098244_rwf.chk
%rwf=calculation/no_transfer_pubchem_sampling_0_b3lyp/45098244_rwf.rwf
%nprocshared=14
%mem=3GB
#p b3lyp/6-31g(d) 6D 10F nosymm GFInput td=(50-50,nstate=2) SCF=XQC 

Gaussian input prepared by ASE

0 1
C                -2.1460530183        1.5829296840       -0.2186173917
C                -1.1489403881        0.4315516445       -0.1827662783
C                -1.6745911637       -0.8510194028        0.5539454475
C                -0.8144911235       -2.0094163103       -0.0373087481
C                 0.1681268485       -1.2669820687       -0.9739200853
C                -0.7637518638       -0.1869932332       -1.5554710659
C                 1.1778848816       -0.4058548009       -0.1462502684
C                 2.2536593409        0.2061872182       -1.0748726754
C                 1.9037916550       -1.1796834326        0.9717252240
C                 0.2358942583        0.6859227851        0.4011020261
C                 0.6500829315        1.6355137602        1.2616039165
C                -0.1460599210        2.6657253239        1.8449240816
N                -0.7454731756        3.5208380821        2.3615870784
H                -2.4605438642        1.8915924642        0.7819121660
H                -1.7277450877        2.4623849721       -0.7201755144
H                -3.0415659819        1.2732000714       -0.7705867274
H                -1.5871603478       -0.7571020200        1.6409189498
H                -2.7381541769       -0.9820377832        0.3232207010
H                -0.3175986539       -2.6106529728        0.7281360873
H                -1.4349640637       -2.6949297665       -0.6248892910
H                 0.6687590703       -1.9187217298       -1.6967239566
H                -0.2769441556        0.5260753456       -2.2275706260
H                -1.6368899444       -0.6042803137       -2.0705287812
H                 2.8943455274        0.9004553681       -0.5201462370
H                 2.8916228291       -0.5876970375       -1.4820641524
H                 1.8225551154        0.7570464684       -1.9153884493
H                 2.4186953078       -2.0549271121        0.5568471165
H                 2.6625908630       -0.5506650535        1.4505147514
H                 1.2233488858       -1.5228628620        1.7556607877
H                 1.6895694153        1.6444017121        1.5851809143


