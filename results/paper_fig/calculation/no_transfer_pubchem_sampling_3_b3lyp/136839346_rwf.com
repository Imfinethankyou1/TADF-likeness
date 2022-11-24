%chk=calculation/no_transfer_pubchem_sampling_3_b3lyp/136839346_rwf.chk
%rwf=calculation/no_transfer_pubchem_sampling_3_b3lyp/136839346_rwf.rwf
%nprocshared=14
%mem=3GB
#p b3lyp/6-31g(d) 6D 10F nosymm GFInput td=(50-50,nstate=2) SCF=XQC 

Gaussian input prepared by ASE

0 1
C                 5.3718988920        0.9132084320        0.9091560399
C                 4.7776099347        1.5738842618        1.9492446971
S                 3.0575113319        1.3442603573        1.9628532431
C                 3.1345067041        0.3336148511        0.5352185714
C                 1.9276005005       -0.2454048505       -0.0346881362
C                 1.8232283354       -1.0822509106       -1.1808082106
C                 0.4708333456       -1.3497962588       -1.2900768585
N                -0.1515374269       -0.6803311227       -0.2727492467
N                 0.7367617214       -0.0161436484        0.5227837983
C                -1.5569558518       -0.7974346504        0.1026018012
C                -2.1684017423        0.5297184892        0.5169675304
C                -3.1320403388        0.5680884951        1.5268431262
C                -3.7443359119        1.7834838997        1.8356757948
N                -3.4604605114        2.9368138219        1.2207008376
C                -2.5278049443        2.8916725046        0.2594727869
C                -1.8612897057        1.7306478952       -0.1287920128
C                -2.3226769757       -1.4053703788       -1.0956576508
C                -1.5599067744       -2.5747613522       -1.7214943074
N                -0.2710475385       -2.0803336028       -2.2118845134
C                 4.4341039997        0.2057841264        0.1021550630
H                 6.4400377743        0.9277801353        0.7200706209
H                 5.2468260169        2.1839323397        2.7096780437
H                 2.6114948344       -1.4297239505       -1.8321343233
H                -1.6253293912       -1.4865728400        0.9559635514
H                -3.4016673160       -0.3320761196        2.0741044586
H                -4.4973923107        1.8312756469        2.6209502947
H                -2.2981239323        3.8385276582       -0.2265218377
H                -1.1028594519        1.7674116199       -0.9038949247
H                -3.3087818006       -1.7321023362       -0.7513847227
H                -2.4748163112       -0.6319157414       -1.8565829386
H                -1.4429306781       -3.3857878813       -0.9818563413
H                -2.1195802673       -2.9758241372       -2.5719979160
H                 0.2848079615       -2.7765091672       -2.6943359946
H                 4.7107198283       -0.3777655856       -0.7695813238


