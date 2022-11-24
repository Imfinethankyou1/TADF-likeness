%chk=./calculation/no_transfer_pubchem_sampling_6_b3lyp/20420137.chk
%nprocshared=7
%mem=6GB
#p b3lyp/6-31g(d) opt SCF=XQC 

Gaussian input prepared by ASE

0 1
C                -5.4911777737        0.9071490985       -1.7326650414
C                -4.1814242196        1.2646807631       -1.0980829121
C                -3.4772398696        2.3924208181       -1.5018735122
C                -2.2832780436        2.7239105587       -0.8895697174
C                -1.7711238283        1.9369895496        0.1285076106
C                -2.4588691604        0.8050727811        0.5449565251
C                -1.9330662997       -0.1399670865        1.6050165503
C                -0.9252735326        0.4700542505        2.5724673924
C                 0.4062967527        0.7642944988        1.9256939534
O                 1.1295630088        1.6724523354        2.2665506949
N                 0.7848666169       -0.1092592937        0.9225183823
C                 2.0174542215        0.1854167458        0.2594820404
C                 1.9774274936        0.6791377237       -1.0358989684
C                 3.1578310617        0.9977575492       -1.6918056933
C                 3.1260897691        1.5541449961       -3.0822472560
C                 4.3625739335        0.8218979351       -1.0234403389
C                 4.4114819639        0.3422671952        0.2780035237
C                 5.7226172179        0.1515516090        0.9763836895
C                 3.2225627587        0.0180171752        0.9184915897
C                 0.0302430298       -1.2071456286        0.5494404460
C                -1.2706437434       -1.3054058295        0.9026961644
C                -2.0759550755       -2.4884417352        0.6295634608
O                -3.2433425950       -2.5755952078        0.9619858558
C                -1.3604258749       -3.6345806424       -0.0578872971
C                -0.2752120191       -3.1276656946       -0.9954674743
C                 0.7306903490       -2.2930729832       -0.2119742521
C                -3.6602510955        0.4845879876       -0.0769485376
H                -6.3086818964        1.3796571073       -1.1871359620
H                -5.5315743697        1.2510244044       -2.7639366347
H                -5.6498410644       -0.1683972128       -1.7093209874
H                -3.8674988215        3.0126864250       -2.2963630365
H                -1.7415208252        3.6039341770       -1.2054116280
H                -0.8414943819        2.2262549524        0.5928497576
H                -2.7894877723       -0.5274233955        2.1661429896
H                -0.7347606230       -0.2369115093        3.3869717488
H                -1.3010065796        1.3935100856        3.0149265592
H                 1.0224688894        0.8187230932       -1.5228354359
H                 4.0427762662        1.3174858075       -3.6170595782
H                 3.0285397703        2.6391967098       -3.0426843001
H                 2.2784466864        1.1615536046       -3.6391450982
H                 5.2849391263        1.0748317311       -1.5270174698
H                 6.4733351923        0.8336784171        0.5853845174
H                 6.0804612139       -0.8677346589        0.8262682702
H                 5.6164021299        0.3178915924        2.0454306059
H                 3.2311482989       -0.3500882044        1.9331462659
H                -2.0983943425       -4.2403424553       -0.5853932787
H                -0.9157873867       -4.2585658276        0.7238683950
H                 0.2308867781       -3.9655400991       -1.4780217375
H                -0.7284878448       -2.5105750023       -1.7746031961
H                 1.4769342263       -1.8661735627       -0.8834777681
H                 1.2631500134       -2.9295432949        0.5044796749
H                -4.1933687294       -0.3998023540        0.2411044480


