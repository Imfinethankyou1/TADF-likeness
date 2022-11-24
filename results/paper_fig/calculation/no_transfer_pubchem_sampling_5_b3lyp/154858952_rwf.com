%chk=calculation/no_transfer_pubchem_sampling_5_b3lyp/154858952_rwf.chk
%rwf=calculation/no_transfer_pubchem_sampling_5_b3lyp/154858952_rwf.rwf
%nprocshared=14
%mem=3GB
#p b3lyp/6-31g(d) 6D 10F nosymm GFInput td=(50-50,nstate=2) SCF=XQC 

Gaussian input prepared by ASE

0 1
C                 1.4747136905        0.5752337701        3.5782174573
C                 2.0036725141        0.6942954195        2.1749996996
N                 3.1020785950       -0.0616270241        1.8266248795
C                 3.4651065706        0.1164759228        0.5860460228
C                 4.6006681987       -0.5687969121       -0.0460809782
C                 5.3386822776       -1.4987182661        0.7081795515
C                 6.4197695647       -2.1662270625        0.1405921047
C                 6.7851356019       -1.9203510118       -1.1857882012
C                 6.0584634152       -0.9983743840       -1.9416505419
C                 4.9754059831       -0.3276371725       -1.3776703190
S                 2.4528760067        1.2786855565       -0.2780010211
C                 1.5025724990        1.4969060533        1.1770752523
C                 0.2996945387        2.4125276382        1.2187695395
C                 0.5289769932        3.7628478301        0.5257764172
N                -0.9004042644        1.7620988382        0.6851797967
C                -1.9079898536        1.3001639329        1.4947742396
O                -1.9080168957        1.4429453046        2.7118032806
C                -3.0004366706        0.5881001426        0.7642820418
C                -2.8620176750       -0.2072283098       -0.3649528849
N                -4.1266353240       -0.6830928963       -0.6362630338
C                -4.5824377118       -1.5292067067       -1.6318455384
C                -3.5759384294       -2.0504023130       -2.6109920484
C                -5.9208686433       -1.8236154770       -1.6454531780
C                -6.8121852392       -1.2852283352       -0.6747684648
C                -6.3485677059       -0.4520008197        0.3114603643
C                -4.9711330989       -0.1340732713        0.3534852948
N                -4.2901548414        0.6278498670        1.2018274114
H                 1.5231940940       -0.4708940636        3.8962516313
H                 0.4386308241        0.9124366485        3.6592790946
H                 2.0900044116        1.1540737784        4.2792403193
H                 5.0458096381       -1.6806905580        1.7364345823
H                 6.9806038388       -2.8819273936        0.7358019713
H                 7.6299531047       -2.4425633507       -1.6265345128
H                 6.3355759418       -0.7988413479       -2.9732196454
H                 4.4217924847        0.3895933255       -1.9784358840
H                 0.0643005601        2.5942269706        2.2697013841
H                 0.6957055189        3.6413598613       -0.5521446097
H                 1.4077357105        4.2641245919        0.9432412960
H                -0.3442390544        4.4075519804        0.6646503631
H                -1.0052133804        1.6842861429       -0.3171783110
H                -1.9972733158       -0.5112024616       -0.9328062589
H                -2.7907233091       -2.6301458955       -2.1079360078
H                -3.0815001153       -1.2320401973       -3.1511598183
H                -4.0610735997       -2.6986221737       -3.3443311272
H                -6.2957647566       -2.4838168133       -2.4203397871
H                -7.8653728077       -1.5437656879       -0.7222427373
H                -6.9931788839       -0.0246956698        1.0708649140


