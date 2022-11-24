%chk=./calculation/no_transfer_pubchem_sampling_5_b3lyp/50746014.chk
%nprocshared=7
%mem=6GB
#p b3lyp/6-31g(d) opt SCF=XQC 

Gaussian input prepared by ASE

0 1
C                 7.7894032656        0.6485116974       -1.3005893478
C                 8.2111739661        0.1578514213        0.0598692865
O                 9.3507572135        0.2354185058        0.4647626410
N                 7.2651572358       -0.4347351254        0.8517266740
C                 5.9034885907       -0.6821812903        0.6631256574
C                 5.3711139935       -1.8650760580        1.1780960167
C                 4.0315874497       -2.1554116249        1.0354224326
C                 3.1819717916       -1.2686758659        0.3767197713
N                 1.8317984783       -1.6249266820        0.2504883324
C                 0.8176410858       -0.9181767016       -0.3214828438
O                 0.9191194618        0.1649461064       -0.8407798466
C                -0.5198413950       -1.6769508284       -0.2353008264
O                -1.5797893169       -0.9565804064       -0.7971107957
C                -1.5846087687       -0.6854568555       -2.1210606809
C                -0.6908520296       -1.2134965398       -3.0479721806
C                -0.7990664286       -0.8976784953       -4.3900097631
C                -1.8004415788       -0.0537101687       -4.8416471325
C                -2.6977446862        0.4763545214       -3.9378311635
C                -2.6052513020        0.1701678774       -2.5816856244
C                -3.5559941602        0.7438613036       -1.6591045062
N                -3.6578257135        0.5484112614       -0.3769656272
C                -4.7111035708        1.3322235687       -0.0113983907
C                -5.2278610707        1.4647161647        1.3393970056
C                -6.3701686474        2.2474566425        1.5164831218
C                -6.9204369411        2.4221052556        2.7690795370
C                -6.3298497276        1.8127632159        3.8645539407
C                -5.1970463691        1.0384204696        3.6942584436
C                -4.6237036280        0.8465882003        2.4429144943
C                -3.3973509283        0.0025346009        2.3111985383
N                -5.2312308264        1.9738023488       -1.0334385236
O                -4.4926876102        1.6024408583       -2.1086358389
C                 3.7008152370       -0.0760033195       -0.1148627336
C                 5.0434995776        0.2148025268        0.0349971950
H                 7.4458966944        1.6785966708       -1.2310913634
H                 7.0008209379        0.0314338914       -1.7215813767
H                 8.6625088906        0.6328191821       -1.9484124626
H                 7.6687366437       -0.8695350350        1.6740063533
H                 6.0186655244       -2.5605261731        1.6917339458
H                 3.6386760593       -3.0785909005        1.4379088827
H                 1.5975413779       -2.5162523987        0.6663390713
H                -0.4150761608       -2.6728402857       -0.6922815392
H                -0.7944181454       -1.7972340303        0.8175463580
H                 0.1089012895       -1.8641629304       -2.7301224139
H                -0.0896591919       -1.3166404475       -5.0880657539
H                -1.8765514396        0.1899221456       -5.8896875358
H                -3.4848941934        1.1378755738       -4.2652842970
H                -6.8090608984        2.7109653050        0.6469096051
H                -7.8045399255        3.0291937193        2.8936914325
H                -6.7510264051        1.9414834836        4.8510561776
H                -4.7421682288        0.5685777011        4.5553148623
H                -2.5910649333        0.5725100131        1.8545541958
H                -3.0779785016       -0.3603591889        3.2861457724
H                -3.5900734189       -0.8426829405        1.6540012106
H                 3.0491774294        0.6283125487       -0.6031735274
H                 5.4109169476        1.1628175110       -0.3186918599


