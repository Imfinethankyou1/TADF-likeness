%chk=calculation/no_transfer_pubchem_sampling_4_b3lyp/19587883_rwf.chk
%rwf=calculation/no_transfer_pubchem_sampling_4_b3lyp/19587883_rwf.rwf
%nprocshared=14
%mem=3GB
#p b3lyp/6-31g(d) 6D 10F nosymm GFInput td=(50-50,nstate=2) SCF=XQC 

Gaussian input prepared by ASE

0 1
C                -7.0758159064       -2.4841968557       -0.2746202891
C                -7.7421728930       -1.8988525537        0.7180805815
C                -7.4415361247       -0.5498825380        1.3265963563
N                -6.3826321537        0.2026169904        0.6682615050
C                -6.4815687334        0.9089541084       -0.5474308944
C                -7.5742181053        1.0172879533       -1.4046169765
C                -7.4150244221        1.7959559784       -2.5522473073
C                -6.2031289193        2.4438553462       -2.8422693924
C                -5.1116429866        2.3275332936       -1.9850206851
C                -5.2509314157        1.5558347060       -0.8289318615
C                -4.3726799102        1.2174021122        0.2689161395
N                -3.1160666476        1.5268277189        0.5288221712
N                -2.5910399210        1.0372417243        1.6672827329
C                -3.3492270271        0.2848073254        2.4834618920
S                -2.6537558573       -0.3405901567        3.9903452838
C                -1.0529042295        0.5591393289        4.1307040171
C                 0.1364421205       -0.1041029490        3.4235336399
O                 0.9759995663       -0.7266933149        4.0577280435
N                 0.1471616378        0.1056152927        2.0681239406
C                 1.0604498701       -0.3635684340        1.1145007261
C                 2.1849609216       -1.1444846255        1.4347425708
C                 3.0417777224       -1.5658345022        0.4220790815
C                 2.7844081106       -1.2104210253       -0.9026622282
S                 3.8635387440       -1.7718344050       -2.2042994297
O                 4.5579470230       -2.9744442661       -1.7526692668
O                 3.1252957257       -1.7446414585       -3.4711318592
N                 4.9790805334       -0.4922922571       -2.4277045372
C                 6.1172030266       -0.2461167031       -1.6649585943
N                 6.1224009157       -0.6746845290       -0.4006715094
C                 7.2240712145       -0.3942199507        0.3113529353
C                 7.2487376310       -0.8907140815        1.7317795639
C                 8.2844026680        0.3237881246       -0.2473516576
C                 8.1649151801        0.7358801407       -1.5780077260
C                 9.2449312784        1.5176877876       -2.2772514842
N                 7.0739129197        0.4441726921       -2.3004166763
C                 1.6655971137       -0.4414547281       -1.2350215041
C                 0.8094841379       -0.0199610735       -0.2278896658
N                -4.6311893455       -0.0869958319        2.2914948192
C                -5.1248832370        0.3858155615        1.1637086045
H                -6.2248989815       -2.0133735323       -0.7588022567
H                -7.3600214006       -3.4671680747       -0.6375447091
H                -8.5890895606       -2.4068167354        1.1791473574
H                -7.1329968516       -0.6715774310        2.3711644966
H                -8.3481796406        0.0702266152        1.3351693874
H                -8.5080688669        0.5042894036       -1.2009577947
H                -8.2504523297        1.8969945887       -3.2391261342
H                -6.1174677497        3.0383871214       -3.7466663388
H                -4.1698423411        2.8222846408       -2.2005847313
H                -0.8308425791        0.5521791500        5.1991715757
H                -1.2087700938        1.5881623496        3.7964628848
H                -0.6669987645        0.6137316067        1.7112046375
H                 2.3686259076       -1.4136325873        2.4652560233
H                 3.9131811742       -2.1658631430        0.6537914967
H                 5.0484926747       -0.2234829803       -3.4043753818
H                 7.1980323120       -1.9854933930        1.7469161777
H                 8.1535883704       -0.5736156528        2.2575220730
H                 6.3726245908       -0.5213214939        2.2758464705
H                 9.1717223053        0.5538454213        0.3325073592
H                 9.5932463153        0.9718423261       -3.1611247821
H                 8.8488998335        2.4767754553       -2.6293821969
H                10.0982284494        1.7075550917       -1.6202965115
H                 1.4654610124       -0.1930751918       -2.2715833645
H                -0.0667760113        0.5747214994       -0.4741947978


