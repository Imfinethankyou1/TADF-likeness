%chk=calculation/no_transfer_pubchem_sampling_2_b3lyp/123904104_rwf.chk
%rwf=calculation/no_transfer_pubchem_sampling_2_b3lyp/123904104_rwf.rwf
%nprocshared=14
%mem=3GB
#p b3lyp/6-31g(d) 6D 10F nosymm GFInput td=(50-50,nstate=2) SCF=XQC 

Gaussian input prepared by ASE

0 1
C                -4.7144065816       -2.3837486021        3.4647535313
O                -4.4940755798       -2.0705366242        2.0825486068
C                -3.4542194843       -1.2341877130        1.8370917669
O                -2.7462558378       -0.7948369384        2.7336206839
C                -3.3911716897       -0.9199002235        0.4089980588
C                -2.4805489234       -0.1670769304       -0.2728279819
C                -2.9524282184        0.1860728013       -1.6815500948
O                -4.1128505848        0.1887966947       -2.0352577891
O                -1.9269304711        0.4699979192       -2.5082076897
C                -2.2972107638        0.8365806174       -3.8481746057
N                -1.2063491905        0.2749822562        0.0757685237
C                -1.0592115014        1.7470966531       -0.0066137559
C                -0.0463651012        2.3819429349       -0.9918115889
C                -0.4543242815        3.6563772494       -1.4256122505
C                 0.2508963290        4.4370852590       -2.3358054153
C                 1.4184087051        3.9217696172       -2.8837135524
C                 1.8436221614        2.6490092252       -2.5159254490
C                 1.1348084851        1.8636965383       -1.5855189635
O                 1.5689855504        0.6039407546       -1.2592154124
C                 2.8345923266        0.1968500570       -1.7907480065
C                 3.2691499797       -1.1621060079       -1.2759561362
O                 3.7250883379       -1.0498573317        0.0662927797
C                 4.0308375936       -2.2728703301        0.7183530475
C                 3.2883311671       -2.4047511593        2.0556510371
O                 1.8718352460       -2.1894737469        2.0170509044
C                 1.1840867147       -2.6259918166        0.9069755783
C                 1.4237706338       -3.9096285317        0.3995697758
C                 0.8408672587       -4.3307090491       -0.7930680482
C                 0.0231074633       -3.4436569465       -1.4872595527
C                -0.2651041354       -2.1933077641       -0.9378141348
C                 0.2570582536       -1.7569234960        0.2893191578
C                -0.3647899043       -0.5314954272        1.0028514906
C                 0.4718878279        0.3345527123        1.9988604309
C                 1.8347719334        0.9666261372        1.6339618258
C                -0.3402304579        1.5136739814        2.4967859843
O                -0.2954783761        1.8796428887        3.6564782729
C                -1.1575232208        2.2998817538        1.4658784602
C                -0.9764637200        3.8098474834        1.6705969185
H                -3.8390848839       -2.8803510860        3.8924834313
H                -4.9193764150       -1.4758980742        4.0390210868
H                -5.5780656822       -3.0495129538        3.4783661876
H                -4.2558274132       -1.2401163089       -0.1587735279
H                -2.8001493440        0.0030448416       -4.3453868016
H                -1.3606367486        1.0811142853       -4.3486209536
H                -2.9664499248        1.7003902269       -3.8365995417
H                -2.0062971200        2.0796967450       -0.4367318296
H                -1.3852804641        4.0516304966       -1.0301889628
H                -0.1195373403        5.4182368207       -2.6173760959
H                 1.9982661712        4.4913458947       -3.6049593650
H                 2.7444662436        2.2637642023       -2.9730111154
H                 3.6049314045        0.9345444168       -1.5323267311
H                 2.7680895530        0.1324605133       -2.8858552515
H                 4.0948031389       -1.4963092980       -1.9264033345
H                 2.4514235386       -1.8835835196       -1.3647423739
H                 3.8082564588       -3.1201368874        0.0627744791
H                 5.1076367319       -2.3070393695        0.9450403898
H                 3.6514094055       -1.6481644679        2.7565271156
H                 3.5219473257       -3.3931175528        2.4803779643
H                 2.0874048838       -4.5720204715        0.9477868487
H                 1.0407478784       -5.3277920897       -1.1760418623
H                -0.4115253350       -3.7227509362       -2.4431684652
H                -0.8970918087       -1.5211620752       -1.4989052918
H                -1.0622650942       -0.9979384333        1.6994437560
H                 0.6498594182       -0.3105162909        2.8593101540
H                 1.7408612637        1.8314861894        0.9721799339
H                 2.2862471512        1.3201389558        2.5678586359
H                 2.4996614631        0.2476562683        1.1657406079
H                -2.1911026136        2.0541953690        1.7526559487
H                -0.0039887550        4.1573080044        1.3088515829
H                -1.7563573848        4.3863174817        1.1625105966
H                -1.0391416461        4.0157112071        2.7418373771


