%chk=./calculation/no_transfer_pubchem_sampling_1_b3lyp/140909933.chk
%nprocshared=7
%mem=6GB
#p b3lyp/6-31g(d) opt SCF=XQC 

Gaussian input prepared by ASE

0 1
C                 4.3213098465        6.1390143998        1.0148393332
N                 3.6809103144        5.1833919306        0.8161746360
C                 2.9393376699        4.0704115963        0.5858473294
C                 1.6903324562        4.1624784572       -0.0302602364
C                 0.9496513597        3.0276504810       -0.2707535665
C                 1.4297775595        1.7650461192        0.0923112407
C                 0.5942424343        0.6137740483       -0.1799485679
N                -0.7108445545        0.6820921604       -0.2651275152
C                -1.1683951129       -0.5801318888       -0.4922570438
C                -2.5774786502       -0.9677438791       -0.7284881412
O                -2.8206994387       -2.0491852352       -1.2565667811
N                -3.5691510249       -0.1142675144       -0.3955696935
C                -3.4651889166        1.0853818683        0.4008777137
C                -4.1053770700        0.8518493846        1.7711818252
C                -5.5578404130        0.4144246652        1.6094449373
C                -5.6447552824       -0.8205599405        0.7096320467
N                -7.0408449900       -1.1188157149        0.4329305194
C                -4.9379949128       -0.5248667425       -0.6197221581
C                -0.0927602484       -1.4500827355       -0.5534665547
C                -0.0334328030       -2.9068284177       -0.8096168441
N                 1.0241652456       -0.6832009603       -0.3429357191
C                 2.3515885868       -1.1297228769       -0.4558509534
C                 3.2036652755       -0.5882197054       -1.4071631785
N                 4.4617787382       -0.9679038874       -1.5499906646
C                 4.9558955268       -1.9152379885       -0.7704083995
O                 6.2206903728       -2.3029934853       -0.9066525316
C                 6.9784477612       -1.6382867603       -1.9026652760
C                 4.1538576282       -2.5289514439        0.2145162628
F                 4.6683665110       -3.4916188832        0.9949738459
C                 2.8474514201       -2.1370735454        0.3783905541
C                 2.6643776668        1.6682798332        0.7340387374
C                 3.4019049328        2.8092077343        0.9664453707
F                 4.5927971564        2.6968728261        1.5813100268
H                 1.3220495547        5.1359568863       -0.3157171002
H                -0.0159679026        3.0905958037       -0.7454161776
H                -2.4196825126        1.3694200025        0.4993260536
H                -4.0025134557        1.8860660967       -0.1202900737
H                -4.0503556227        1.7693028798        2.3584541456
H                -3.5448999218        0.0760000959        2.2979038334
H                -6.1419651975        1.2131332963        1.1488153162
H                -5.9986387008        0.1948992623        2.5842442228
H                -5.1119029909       -1.6571405490        1.2007279613
H                -7.5341507972       -1.3328368094        1.2928580227
H                -7.1146134871       -1.9252875959       -0.1776401207
H                -5.4774518802        0.2758517966       -1.1347202558
H                -4.9253125334       -1.4151372707       -1.2508549376
H                 0.0933134379       -3.4631654075        0.1195258831
H                -0.9722212307       -3.2100052357       -1.2655944866
H                 0.7969086163       -3.1483744818       -1.4720777383
H                 2.8688833340        0.1875706956       -2.0815106398
H                 7.9700400953       -2.0825481472       -1.8648623112
H                 6.5293935177       -1.7839934387       -2.8869190515
H                 7.0295226099       -0.5679858833       -1.6941320322
H                 2.2352266009       -2.5923136735        1.1407210278
H                 3.0585534210        0.7258057779        1.0779099046


