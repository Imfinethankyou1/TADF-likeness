%chk=./calculation/no_transfer_pubchem_sampling_5_b3lyp/117629897.chk
%nprocshared=7
%mem=6GB
#p b3lyp/6-31g(d) opt SCF=XQC 

Gaussian input prepared by ASE

0 1
O                 2.3633465536       -1.9740205557       -0.9951289788
C                 1.7909523802       -0.9804643544       -0.5968530726
N                 0.5567554790       -0.5773871895       -1.0318349578
C                -0.3281657133       -1.2225265820       -1.8904566191
C                 0.0331402773       -2.3267000274       -2.6573524997
C                -0.8774732998       -2.9108458204       -3.5175704338
C                -2.1613981240       -2.4057729279       -3.6363241307
C                -2.5385509241       -1.3142390247       -2.8758024818
C                -1.6386947272       -0.7252248770       -1.9993879165
N                -2.0450222660        0.4011673810       -1.2551645684
C                -2.5787087141        1.5521681291       -1.7684098610
N                -2.8212839963        2.4649705224       -0.8845532176
C                -2.4499027295        1.9183596773        0.3185440044
C                -2.4873742682        2.4515122094        1.6021974914
C                -2.0434637823        1.6584674625        2.6386419687
C                -1.5862485249        0.3563062466        2.4233163565
C                -1.5420365248       -0.1962652699        1.1599989678
C                -1.9593744556        0.6074061891        0.1074157100
C                 2.3633117778       -0.1025144465        0.4615462891
C                 3.3282674986       -0.6700101501        1.2912265882
C                 3.8944677789        0.0512739972        2.3207171262
C                 3.5109721210        1.3716396348        2.5356621928
O                 4.0314254717        2.1235703551        3.5341471629
C                 2.5646747790        1.9559835487        1.6958893254
C                 1.9980759098        1.2271561677        0.6722372337
H                 0.1788279192        0.2403077019       -0.5740054802
H                 1.0341870341       -2.7139678161       -2.5691931781
H                -0.5771649730       -3.7656057671       -4.1037784751
H                -2.8686314432       -2.8595666588       -4.3129412926
H                -3.5384618053       -0.9117901287       -2.9459263649
H                -2.7491649548        1.6683193978       -2.8215822270
H                -2.8566675870        3.4507008132        1.7660499027
H                -2.0560843324        2.0448622045        3.6466837734
H                -1.2625359856       -0.2340665152        3.2671088822
H                -1.2023086945       -1.2088793006        0.9998168978
H                 3.6199721406       -1.6935817783        1.1113882870
H                 4.6369787770       -0.4052679254        2.9616183491
H                 4.6806850552        1.6121045891        4.0329963167
H                 2.2899334725        2.9848845526        1.8634973510
H                 1.2927424005        1.7298503359        0.0255625789


