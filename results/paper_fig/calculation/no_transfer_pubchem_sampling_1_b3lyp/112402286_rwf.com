%chk=calculation/no_transfer_pubchem_sampling_1_b3lyp/112402286_rwf.chk
%rwf=calculation/no_transfer_pubchem_sampling_1_b3lyp/112402286_rwf.rwf
%nprocshared=14
%mem=3GB
#p b3lyp/6-31g(d) 6D 10F nosymm GFInput td=(50-50,nstate=2) SCF=XQC 

Gaussian input prepared by ASE

0 1
C                -1.9471602228       -0.2021190689        3.5269063060
C                -0.9965413101       -1.3653717415        3.2202849053
N                 0.0715955460       -1.0134724510        2.2841987669
C                 0.1259759820       -1.3780179060        0.9479906725
N                 1.1913368272       -1.4475610421        0.2174186740
C                 2.4873728989       -1.2714476844        0.8374373239
C                 2.9556428433        0.2066932919        0.9032848274
O                 3.9578059448        0.3687028252        1.9122130546
C                 3.4161012532        0.7863624243       -0.4301306014
C                 2.9666187788        0.3039012913       -1.6653222542
C                 3.4411021501        0.8989387931       -2.8353428008
N                 4.3104643182        1.9171783480       -2.8651199011
C                 4.7337374355        2.3730252752       -1.6790589025
C                 4.3250323240        1.8515472886       -0.4521535963
N                -1.1100039278       -1.7109508831        0.4046888871
C                -1.2351762773       -1.9960965455       -1.0193438871
C                -1.3015276405       -0.7389663513       -1.9176133596
C                -2.4766454507        0.1547245934       -1.5960355162
C                -2.3465449505        1.3324029053       -0.8468169932
C                -3.4529699178        2.1264094769       -0.5329781863
C                -4.7250273164        1.7531906262       -0.9692013354
C                -4.8893575548        0.5852665465       -1.7167497557
C                -3.7697468381       -0.1823994974       -2.0068338630
F                -3.9366932433       -1.3211062866       -2.7255543575
H                -2.7201049498       -0.5145769111        4.2390193461
H                -1.4014069536        0.6420663849        3.9625213240
H                -2.4464260099        0.1605122045        2.6206404621
H                -1.5495162908       -2.2172434741        2.8121454366
H                -0.5304201793       -1.7107983260        4.1511085692
H                 0.9590288715       -0.7739942145        2.6988095726
H                 3.2274366991       -1.8451052752        0.2616714916
H                 2.5400982626       -1.6566116368        1.8708680466
H                 2.1136960454        0.8094919881        1.2681756255
H                 4.7606922206       -0.0770083169        1.5947172748
H                 2.2555699839       -0.5147658240       -1.7074678500
H                 3.1042356466        0.5329038640       -3.8044992604
H                 5.4405513052        3.2014303491       -1.7107563736
H                 4.7077698251        2.2616105693        0.4768146376
H                -1.8865368383       -1.1823453141        0.7831928095
H                -2.1416638815       -2.5942039388       -1.1617648328
H                -0.3735384388       -2.6005259971       -1.3063519636
H                -0.3690722227       -0.1787852109       -1.8014054928
H                -1.3611230183       -1.0720539258       -2.9608406736
H                -1.3543857942        1.6318764307       -0.5178945254
H                -3.3188122087        3.0369214949        0.0439286411
H                -5.5897325511        2.3668984590       -0.7333422895
H                -5.8617321750        0.2634733930       -2.0754560832


