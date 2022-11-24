%chk=./calculation/no_transfer_pubchem_sampling_4_b3lyp/57170756.chk
%nprocshared=7
%mem=6GB
#p b3lyp/6-31g(d) opt SCF=XQC 

Gaussian input prepared by ASE

0 1
C                -0.9555544152       -6.6118681079        0.1950246818
C                 0.5272612510       -6.9689355644        0.1999441640
O                 1.3630446121       -6.0112483285        0.8190896674
C                 1.5162814085       -4.7997575505        0.1234452849
C                 2.7846617546       -4.1046707066        0.6740663179
C                 3.0832298979       -2.8291710915       -0.0597756159
C                 3.8044783444       -2.8516780710       -1.2430482015
C                 4.0709268232       -1.6915217683       -1.9500607786
C                 3.6060315627       -0.4689277979       -1.4772380770
O                 3.7752216504        0.7364369275       -2.0855637239
C                 4.5971983017        0.8408030040       -3.2208460791
C                 6.0919232659        0.7535578451       -2.8989191887
N                 6.4942271561        1.8555139049       -2.0540725672
C                 7.9078386158        1.8566308113       -1.7474546681
C                 8.3329551617        3.1948131248       -1.2609884955
N                 7.9140245524        4.3264319457       -1.7125161853
C                 8.5928649382        5.2788227752       -0.9850839407
C                 8.5614643957        6.6654907640       -1.0203354012
C                 9.3894866395        7.3362002637       -0.1425239061
C                10.2251625734        6.6586963281        0.7450737509
C                10.2699160025        5.2771184581        0.7942446325
C                 9.4410201692        4.6055083942       -0.0831633426
O                 9.2605181154        3.2710469996       -0.2748959182
C                 2.8738335864       -0.4432225112       -0.2886299246
C                 2.6164780378       -1.6039689538        0.4063727751
C                 0.3703279793       -3.8252615471        0.3416416165
O                 0.0026193092       -2.9903274643       -0.4284930014
O                -0.2062297415       -4.0383750807        1.5567370982
C                -0.7857110391       -3.0664726411        2.3144652505
O                -0.4103784408       -1.9334371340        2.3392863760
C                -1.9500031355       -3.6378246934        3.1077655330
C                -3.2028077860       -3.4667033797        2.2153243389
C                -3.4819305237       -2.0210288999        1.9210982677
C                -4.2166902893       -1.2525901687        2.8105227020
C                -4.4662760515        0.0882903410        2.5734081727
C                -3.9670962628        0.6922492506        1.4244860415
O                -4.1259975896        2.0009521974        1.0812132177
C                -4.9610535133        2.8322889633        1.8547271497
C                -6.4581245658        2.5604852440        1.6458811693
N                -6.9331928873        3.2099218126        0.4401612905
C                -6.4329428816        2.6018349773       -0.7813343033
C                -7.1008807847        3.1911185964       -1.9696353376
N                -6.5304631142        3.5548733940       -3.0632515011
C                -7.5487818142        4.0098188904       -3.8714057665
C                -7.5478885515        4.5266691369       -5.1590546442
C                -8.7667533509        4.9003477119       -5.6891929347
C                -9.9541926120        4.7679493570       -4.9699322076
C                -9.9750984433        4.2548616953       -3.6855188468
C                -8.7558215408        3.8807303626       -3.1554962555
O                -8.4534007776        3.3504127962       -1.9414204048
C                -3.2234083929       -0.0767938225        0.5274818767
C                -2.9841628336       -1.4106735767        0.7739585164
O                -1.8190232716       -5.0075617374        3.3922126503
C                -0.9999829766       -5.3255325945        4.5002014766
C                 0.4892072556       -5.0777951121        4.2834098560
H                -1.5272347906       -7.4609260264       -0.1710333001
H                -1.1470457071       -5.7616684771       -0.4575487720
H                -1.2865554853       -6.3662937962        1.2015044680
H                 0.6887354000       -7.8784864821        0.7841463514
H                 0.8725157588       -7.1456377316       -0.8307668664
H                 1.6265104677       -4.9695620808       -0.9591501854
H                 2.6284733469       -3.9167602789        1.7376919791
H                 3.6064901776       -4.8154269111        0.5704822006
H                 4.1735010234       -3.7935284000       -1.6249418062
H                 4.6330929960       -1.7622382212       -2.8683694431
H                 4.3327412966        0.0856105100       -3.9726964483
H                 4.3731915135        1.8302117866       -3.6303578493
H                 6.6505581036        0.7191428594       -3.8543425761
H                 6.2991119954       -0.1709731059       -2.3506885300
H                 6.2412339303        2.7459075328       -2.4721029499
H                 8.5296488711        1.6062237659       -2.6320498238
H                 8.1166659463        1.1096425594       -0.9734717079
H                 7.9140799582        7.1849520977       -1.7073802271
H                 9.3933476932        8.4159504449       -0.1394579930
H                10.8541602124        7.2281753235        1.4123430420
H                10.9132224868        4.7479860983        1.4784979952
H                 2.5098994654        0.5074508762        0.0654361517
H                 2.0404411879       -1.5544285391        1.3186290311
H                -2.0669700683       -3.0456375561        4.0290698677
H                -3.0376707284       -4.0280598563        1.2940861999
H                -4.0378307159       -3.9244251196        2.7486497060
H                -4.6108466049       -1.7066951397        3.7090022895
H                -5.0439328206        0.6452902028        3.2947992895
H                -4.7060780517        2.7541558134        2.9181667007
H                -4.7417577099        3.8470351615        1.5146741488
H                -6.6252147417        1.4688848811        1.6285181630
H                -7.0087645682        2.9853238703        2.4880106632
H                -7.9488831639        3.2044860969        0.4225346862
H                -6.5980221673        1.5056153281       -0.7991000255
H                -5.3591585246        2.7783837759       -0.8647483567
H                -6.6277613457        4.6274094841       -5.7103987097
H                -8.8068200904        5.3073591647       -6.6885455104
H               -10.8826062545        5.0742227887       -5.4277312644
H               -10.8903432613        4.1504684096       -3.1252454075
H                -2.8340315587        0.4018148890       -0.3565093814
H                -2.3980849425       -1.9785889946        0.0667673158
H                -1.1795294737       -6.3910978660        4.6646572792
H                -1.3470408104       -4.7687529525        5.3846658059
H                 1.0458737747       -5.4800561633        5.1259683996
H                 0.8221480498       -5.5653046501        3.3700135867
H                 0.6981871564       -4.0116833415        4.2112711567


