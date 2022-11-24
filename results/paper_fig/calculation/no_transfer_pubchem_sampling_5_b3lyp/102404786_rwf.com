%chk=calculation/no_transfer_pubchem_sampling_5_b3lyp/102404786_rwf.chk
%rwf=calculation/no_transfer_pubchem_sampling_5_b3lyp/102404786_rwf.rwf
%nprocshared=14
%mem=3GB
#p b3lyp/6-31g(d) 6D 10F nosymm GFInput td=(50-50,nstate=2) SCF=XQC 

Gaussian input prepared by ASE

0 1
C                 2.9138504193        1.3847579388        3.6542344780
C                 3.9418841377        0.4899871926        3.3527880249
C                 4.2109157244        0.1536214074        2.0259807559
C                 3.4572552097        0.6990449244        0.9731409231
C                 3.8091682201        0.3809404071       -0.4388631275
C                 5.1564443362        0.4163920596       -0.8364530415
C                 5.5494808619        0.1475627946       -2.1449767388
C                 4.5826443242       -0.1646707917       -3.1014233064
C                 3.2403817943       -0.2144496222       -2.7380413409
C                 2.8450008850        0.0406737880       -1.4161369402
N                 1.4643161750        0.0394807264       -1.1214280424
C                 0.7931745725       -1.0145445950       -0.8897520976
S                 1.6096700750       -2.6157832482       -0.7082906193
C                 0.3467447465       -3.8704573885       -1.1635190999
C                -1.0059916448       -3.6518967108       -0.5156255584
S                -1.8523220093       -2.1436562978       -1.1315164204
C                -0.6932211230       -0.8471036070       -0.6388469788
N                -1.0828801126        0.2693915087       -0.1823111311
C                -2.4297173300        0.5926917172        0.0884591269
C                -3.0092232375        0.2082910731        1.3222351920
C                -2.2864603645       -0.6274045597        2.3230253648
C                -2.9244295501       -1.7454552445        2.8873658479
C                -2.2857901598       -2.5261189067        3.8518315612
C                -0.9943745688       -2.2034359940        4.2722748066
C                -0.3484589412       -1.0950504726        3.7200210107
C                -0.9868750775       -0.3158727468        2.7560457451
C                -4.3267833088        0.6041296253        1.5915860367
C                -5.0572417273        1.3607281214        0.6803480248
C                -4.4734354630        1.7313258622       -0.5264882981
C                -3.1582618653        1.3631785429       -0.8491603809
C                -2.5977749143        1.7665976234       -2.1712800283
C                -1.3062120415        2.3047311037       -2.3055404068
C                -0.8332273953        2.7185100219       -3.5500514001
C                -1.6345725394        2.6016309735       -4.6875291217
C                -2.9163586978        2.0627960179       -4.5705726084
C                -3.3912001614        1.6503109588       -3.3259053240
C                 2.4173040674        1.5895192167        1.2909563714
C                 2.1538404974        1.9318854640        2.6169173650
H                 2.7075533655        1.6560918609        4.6862951546
H                 4.5361565180        0.0514125723        4.1503720121
H                 5.0052872101       -0.5525820958        1.7999416554
H                 5.9037783554        0.6891800334       -0.0965192355
H                 6.5998341578        0.1970437524       -2.4176552546
H                 4.8691427914       -0.3679113693       -4.1297800115
H                 2.4754196501       -0.4534603694       -3.4706036639
H                 0.2544673452       -3.9115867826       -2.2524414309
H                 0.7858873215       -4.8145900373       -0.8205800413
H                -1.6866528784       -4.4697491413       -0.7779922557
H                -0.9326849069       -3.6046986899        0.5745088850
H                -3.9233490618       -2.0110979008        2.5519281372
H                -2.7979349783       -3.3884911463        4.2714515518
H                -0.4947397247       -2.8113283502        5.0221094323
H                 0.6572024965       -0.8306109343        4.0353196795
H                -0.4699435651        0.5393292344        2.3341390333
H                -4.7676634021        0.3246198572        2.5442333606
H                -6.0706486194        1.6728065330        0.9162558152
H                -5.0297837005        2.3394777520       -1.2337078221
H                -0.6688571362        2.3927285023       -1.4332558368
H                 0.1674591189        3.1353162718       -3.6285587832
H                -1.2612954537        2.9228343688       -5.6564706070
H                -3.5472375272        1.9560562388       -5.4493802088
H                -4.3833374586        1.2141149842       -3.2471279814
H                 1.8181721359        2.0164951637        0.4941312009
H                 1.3565011337        2.6363208085        2.8398875908


