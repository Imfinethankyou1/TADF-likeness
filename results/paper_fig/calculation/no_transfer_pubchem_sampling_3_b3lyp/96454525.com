%chk=./calculation/no_transfer_pubchem_sampling_3_b3lyp/96454525.chk
%nprocshared=7
%mem=6GB
#p b3lyp/6-31g(d) opt SCF=XQC 

Gaussian input prepared by ASE

0 1
C                -2.8263905231       -2.5434195045        0.8565887904
C                -3.1522098100       -1.0533452719        0.9217366434
C                -2.2234982137       -0.2766780715        1.8840726776
C                -1.4477737617        0.6319056059        0.9768358978
C                -0.4083004452        1.4807099191        1.2486581432
C                 0.1682899851        2.1733551929        0.1861086781
S                 1.6446939593        3.1156409516        0.4861271641
O                 1.5986989126        4.4144514937       -0.1152885768
O                 2.0194284468        3.0046208215        1.8597394961
N                 2.8142671476        2.2880480933       -0.4663074641
C                 2.6524562886        0.9510788222       -0.7371194993
C                 2.9457365530        0.4018977431       -1.9887702793
C                 2.6616889434       -0.9244249313       -2.2635442519
C                 2.0351371379       -1.7184254106       -1.3212584698
C                 1.7779110843       -1.1772156927       -0.0657784236
N                 1.0639774540       -1.9581327904        0.9043662189
O                 0.5478666998       -2.9916261126        0.5232169410
O                 0.9673469580       -1.5300337700        2.0343500540
C                 2.1274059966        0.1243949492        0.2555980947
C                -0.2954006778        2.0417591796       -1.1095014438
C                -1.3486285615        1.1882410609       -1.3926279876
C                -1.9145980544        0.4709745772       -0.3420371576
N                -2.9506715282       -0.4004858139       -0.3671653194
H                -1.7886145855       -2.7051009808        0.5738719523
H                -3.4650017207       -3.0421243110        0.1300559461
H                -2.9933562685       -3.0000080351        1.8282584730
H                -4.2051086716       -0.9345672648        1.2084206107
H                -1.5538655318       -0.9458219119        2.4266721830
H                -2.7913509668        0.2950639875        2.6184304856
H                -0.0126064822        1.6118468684        2.2426958004
H                 3.1526802765        2.8618210118       -1.2235883302
H                 3.3879123221        1.0288570273       -2.7488736605
H                 2.8996985769       -1.3304152271       -3.2344007922
H                 1.7505249941       -2.7363062337       -1.5336076245
H                 1.9726418437        0.5029660225        1.2518320402
H                 0.1775813955        2.6110484384       -1.8943909332
H                -1.7138167848        1.0743894981       -2.4010393002
H                -3.2747543884       -0.8374039304       -1.2123357767


