%chk=./calculation/no_transfer_pubchem_sampling_3_b3lyp/68498281.chk
%nprocshared=7
%mem=6GB
#p b3lyp/6-31g(d) opt SCF=XQC 

Gaussian input prepared by ASE

0 1
C                 6.4224764129        1.9928598266       -2.9916322424
C                 5.4030727767        1.3290995760       -2.1130568026
C                 4.0560138784        1.5255109883       -2.3330890548
C                 3.1238076949        0.8937469674       -1.5002264618
O                 1.8270344493        1.1471402314       -1.7442354229
C                 0.8240698542        0.4853560017       -1.0503970291
C                 0.8722513904       -0.8976193741       -0.8179627458
C                -0.1653843786       -1.5451924585       -0.1894537056
C                -1.2683201740       -0.7926389067        0.2037438812
C                -1.3333994463        0.6007332634       -0.0502540475
C                -0.2673581533        1.2425092789       -0.6816301357
C                -2.5851829294        1.0369413719        0.4430861813
C                -3.2237910466       -0.0642252048        0.9684290221
C                -4.5079540544       -0.1252231147        1.6961715679
O                -4.6510678907       -0.8025051460        2.7070790891
N                -5.5254823192        0.6332762449        1.2253853068
C                -6.7626126786        0.6423810137        1.9883609364
C                -7.5853535085        1.8635520825        1.5930985477
N                -7.9483588081        1.8704716775        0.1833727773
C                -9.3318325894        1.5324811327       -0.1188866015
C                -9.7146038579        0.1088504442        0.1991977779
C               -10.1821686918       -0.2651476332        1.4554060405
C               -10.4339933698       -1.6009764198        1.7184446889
C               -10.2015262205       -2.5250480027        0.7105745389
N                -9.7808559459       -2.1849885448       -0.4948144778
C                -9.5537968928       -0.9073055303       -0.7406750123
C                -6.9840582050        1.1564851681       -0.6375000687
C                -5.5816221088        1.3023575059       -0.0555924136
N                -2.4137945402       -1.1668201274        0.8402144575
C                -2.7149695613       -2.5093570934        1.2551546745
N                 3.4854547834        0.1109419529       -0.4990516398
C                 4.7723287749       -0.0859184129       -0.2777928433
C                 5.7781404623        0.4895511062       -1.0459031799
N                 7.1394564452        0.2639322721       -0.8086512293
C                 7.7191657784       -0.4529494594        0.1922791168
O                 7.1248947039       -1.0266874692        1.0794721469
C                 9.2182630104       -0.4629833358        0.1220703744
C                 9.9197037266       -0.4948511409       -1.0796099776
C                11.3019771287       -0.5304775334       -1.0782874014
C                11.9874762842       -0.5260266948        0.1292007697
C                13.4891321821       -0.5823875021        0.1324664832
F                14.0299204575        0.0561229660        1.1809776113
F                14.0322563441       -0.0369138350       -0.9673441970
F                13.9469483478       -1.8484191338        0.1868466465
C                11.2981187778       -0.5174457401        1.3354203221
C                 9.9173193586       -0.4928238325        1.3254208774
H                 5.9258597734        2.6085511569       -3.7363392677
H                 7.0831929529        2.6348785801       -2.4079893866
H                 7.0272529879        1.2519467311       -3.5159648903
H                 3.7019296950        2.1572326077       -3.1318307943
H                 1.7400157136       -1.4511303382       -1.1362947007
H                -0.1190546582       -2.6087539003       -0.0145998296
H                -0.2885702456        2.3015493845       -0.8840314614
H                -2.9454483836        2.0457247111        0.4467716090
H                -7.3289691821       -0.2800433052        1.8030541273
H                -6.5062727723        0.6645563536        3.0505231307
H                -8.4842248840        1.9148986482        2.2066519569
H                -6.9954170026        2.7660839109        1.7961740363
H                -9.4776182505        1.7223535970       -1.1865495366
H                -9.9647462028        2.2252501048        0.4422636816
H               -10.3575381756        0.4813675994        2.2179569271
H               -10.8004785023       -1.9171436032        2.6827008721
H               -10.3672940515       -3.5825274498        0.8676823301
H                -9.2386519052       -0.6797553400       -1.7521523077
H                -7.2131364158        0.0812337026       -0.7176391262
H                -7.0113856928        1.5855191547       -1.6445683719
H                -5.3482584670        2.3681091891        0.0529260020
H                -4.8507538159        0.8542037049       -0.7311753631
H                -1.8354226455       -2.9507125229        1.7242385833
H                -3.0064765225       -3.1252177703        0.3991985220
H                -3.5253915594       -2.4859121341        1.9805311057
H                 5.0088148941       -0.7321820256        0.5527813977
H                 7.7605742848        0.7449001415       -1.4425398490
H                 9.3973139809       -0.5275711368       -2.0246768421
H                11.8493409274       -0.5593741631       -2.0080405498
H                11.8417019792       -0.5289048328        2.2674413724
H                 9.3573154942       -0.4986161813        2.2476694768


