%chk=./calculation/no_transfer_pubchem_sampling_0_b3lyp/118033531.chk
%nprocshared=7
%mem=6GB
#p b3lyp/6-31g(d) opt SCF=XQC 

Gaussian input prepared by ASE

0 1
C                -6.5127776300        0.0615735500        2.0743150793
C                -6.1655671259        1.2600949036        1.1919409392
C                -6.4597742385        2.5753449424        1.9116885478
C                -6.9307083794        1.1906237169       -0.1310624776
O                -4.7749858181        1.2028592764        0.7922947779
C                -3.8004534515        1.2611567332        1.7155831863
O                -3.9503909186        1.3225056323        2.9111688438
N                -2.5705667333        1.2941734782        1.1307629408
C                -2.3316323977        1.0176074153       -0.2621652609
C                -2.2284259494       -0.4972602986       -0.5336414520
C                -2.0790832312       -0.7876520876       -2.0320015501
N                -0.8396830063       -0.1919899664       -2.4715282834
C                 0.2628536719       -0.4458590653       -1.6761192069
C                 1.5642377596       -0.2718978271       -2.1162486689
C                 2.6357955198       -0.4955938415       -1.2475594043
N                 3.9208977642       -0.3023508186       -1.7507007906
C                 5.1713196870       -0.4298702832       -1.2024929839
O                 6.1475493091       -0.1969482471       -1.8768944913
O                 5.1526441970       -0.8138507309        0.0693405731
C                 6.3911901236       -0.9981433931        0.8043797138
C                 7.2330988867       -2.0935789859        0.1525325896
C                 5.9278933761       -1.4312067830        2.1956686200
C                 7.1605130854        0.3194834439        0.8794760493
C                 2.3346563786       -0.9027023922        0.0491510530
N                 1.0930357796       -1.0621176411        0.4784894215
C                 0.0844620370       -0.8444585309       -0.3335013255
O                -1.1532857766       -1.0542427722        0.1959541833
H                -7.5744678548        0.0713020098        2.3039980538
H                -5.9487257990        0.1060038781        3.0016051570
H                -6.2711423551       -0.8659591914        1.5591713112
H                -7.5227337058        2.6465913028        2.1241815502
H                -6.1661548453        3.4168916645        1.2876635531
H                -5.9072637565        2.6212435053        2.8458003577
H                -6.6884900873        0.2692235864       -0.6545143224
H                -6.6579471768        2.0306361676       -0.7645838924
H                -7.9995848559        1.2210798699        0.0567887406
H                -1.8037667171        1.1208722529        1.7645887937
H                -3.1554476623        1.4337572480       -0.8426400811
H                -1.3972748642        1.5016823783       -0.5560441630
H                -3.1231890208       -0.9906515135       -0.1400005448
H                -2.9087597551       -0.3445001031       -2.5854301962
H                -2.0936132069       -1.8787086796       -2.1807173655
H                -0.6713569972       -0.1926842744       -3.4668798530
H                 1.7434000388        0.0339184900       -3.1385584825
H                 3.9617138126       -0.0064132674       -2.7179980448
H                 7.5507812530       -1.7836730348       -0.8391707345
H                 6.6522980415       -3.0096140018        0.0669564166
H                 8.1119787987       -2.2885141359        0.7604273461
H                 6.7880531580       -1.5976207705        2.8371074793
H                 5.3522073515       -2.3506852941        2.1288144382
H                 5.2999597805       -0.6616563541        2.6369951491
H                 8.0376079371        0.1944305308        1.5079786196
H                 7.4777549587        0.6267892187       -0.1131840393
H                 6.5289745823        1.0967645283        1.3045909962
H                 3.1023770281       -1.1062044382        0.7750551337


