%chk=./calculation/no_transfer_pubchem_sampling_2_b3lyp/107468036.chk
%nprocshared=7
%mem=6GB
#p b3lyp/6-31g(d) opt SCF=XQC 

Gaussian input prepared by ASE

0 1
C                -3.2148243566       -0.8721915552       -2.6563989037
O                -2.3452671142       -0.4672494764       -1.6261730079
C                -2.8844850188        0.1214984739       -0.5149827234
C                -4.2341273944        0.3501066501       -0.3040867956
C                -4.6662216551        0.9585718906        0.8683889595
C                -3.7613295128        1.3744629250        1.8187471846
C                -2.3946184641        1.1646332591        1.6132587497
C                -1.4753818911        1.7143478376        2.5374504928
N                -0.7731746623        2.2637498197        3.2579272440
C                -1.9496766803        0.5111038723        0.4621716560
N                -0.5981163493        0.2517452225        0.2386881127
C                 0.2488164012       -0.2708600780        1.1913388674
O                -0.1042010196       -0.5313292611        2.3183409425
C                 1.6264681560       -0.5134390747        0.6883417445
C                 2.2405292476       -1.7220476378        1.0127795620
C                 3.5197081386       -1.9789339610        0.5645410490
C                 4.1617023610       -1.0059531340       -0.1999442981
C                 5.5372666398       -1.2235842113       -0.7452498510
N                 3.5939219137        0.1605364194       -0.4628375824
C                 2.3697655749        0.4284576312       -0.0337689153
C                 1.8679448893        1.8066169680       -0.3395765575
H                -3.9202772043       -1.6327570982       -2.3087963658
H                -2.5763378839       -1.2999076204       -3.4276748541
H                -3.7662395423       -0.0231150494       -3.0718621721
H                -4.9632786548        0.0487933342       -1.0389268491
H                -5.7227006564        1.1144194948        1.0224575565
H                -4.0872025297        1.8706323804        2.7195836284
H                -0.3435046229        0.1320679816       -0.7318537495
H                 1.6982430417       -2.4406140053        1.6098981598
H                 4.0168446108       -2.9076632199        0.7978888850
H                 5.5592803457       -2.1113190932       -1.3749967414
H                 5.8391580218       -0.3589521938       -1.3282477217
H                 6.2442733796       -1.3710743052        0.0694058244
H                 1.2282799656        2.1780347307        0.4569796221
H                 2.7221136396        2.4671575469       -0.4622828774
H                 1.3066478859        1.8140545367       -1.2737682751


