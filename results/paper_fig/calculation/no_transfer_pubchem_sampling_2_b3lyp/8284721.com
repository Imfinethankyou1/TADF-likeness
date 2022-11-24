%chk=./calculation/no_transfer_pubchem_sampling_2_b3lyp/8284721.chk
%nprocshared=7
%mem=6GB
#p b3lyp/6-31g(d) opt SCF=XQC 

Gaussian input prepared by ASE

0 1
C                 5.9729531858        2.0227665152        2.5230854604
O                 6.1283652480        2.1304467852        1.1307715123
C                 5.1335791406        1.6503513751        0.3326423414
C                 3.9730932362        1.0417055794        0.7772871553
C                 3.0183440224        0.5870794606       -0.1217235091
C                 3.2177958263        0.7305487584       -1.4819089135
C                 2.1871253576        0.2410396612       -2.4650379052
N                 1.6492213396       -1.0506458603       -2.0869908419
C                 0.3418233443       -1.2356233451       -1.8376408021
O                -0.5113880876       -0.3955866285       -2.0253006662
C                -0.0237153761       -2.6590989266       -1.3737069095
C                -0.6125840260       -3.4478816952       -2.5344034424
O                -1.0194343127       -2.5760186965       -0.3529322079
C                -0.6965505569       -1.8474642276        0.7016584138
O                 0.3836109121       -1.3289536099        0.8562469379
C                -1.7966328730       -1.7882779195        1.7542203661
O                -3.0803645253       -2.0922707704        1.2749924696
C                -3.9255475442       -0.9945219868        1.1039796492
C                -4.4838606438       -0.3701208927        2.2221619472
C                -4.1899700996       -0.8570214048        3.6118132179
C                -5.3397614269        0.7034997499        2.0294196307
C                -5.6613037488        1.1574032841        0.7585609702
C                -6.5699742689        2.3364542779        0.5791289062
C                -5.1199416177        0.4990541629       -0.3340211005
C                -4.2544902689       -0.5757505189       -0.1853900732
C                -3.6844029805       -1.2638513079       -1.3844374728
C                 4.3850757873        1.3373229147       -1.9409848207
C                 5.3408585751        1.8035180793       -1.0566008657
O                 6.5072035876        2.4126371093       -1.4121706223
C                 6.7645561462        2.6171083700       -2.7775718293
H                 5.9039902770        0.9773390634        2.8426783722
H                 6.8668516217        2.4716938013        2.9533900354
H                 5.0898687248        2.5670108112        2.8743130527
H                 3.7909434302        0.9142020594        1.8318956983
H                 2.1204481528        0.1198084451        0.2515727055
H                 2.6225432279        0.1712858161       -3.4683025485
H                 1.3338251907        0.9262638215       -2.5030652586
H                 2.3190991518       -1.7424739151       -1.7897285933
H                 0.8594509566       -3.1608236812       -0.9554531097
H                -1.4511841209       -2.8982832881       -2.9532560244
H                -0.9638964885       -4.4145953346       -2.1838268209
H                 0.1425994897       -3.5932106339       -3.3021072551
H                -1.7486202759       -0.7975180692        2.2192392743
H                -1.5318883737       -2.5371550942        2.5150749874
H                -3.3376766458       -0.3269688017        4.0392752463
H                -3.9642580998       -1.9208113997        3.6015557645
H                -5.0476797180       -0.6847113754        4.2580603508
H                -5.7719029959        1.1893006980        2.8932660491
H                -7.3694432690        2.3244889436        1.3170081263
H                -7.0075555674        2.3416889869       -0.4159805080
H                -6.0085171586        3.2623149140        0.7081882254
H                -5.3721966957        0.8276561166       -1.3323215008
H                -4.3625567282       -1.1815660743       -2.2308428088
H                -2.7356304649       -0.8008249022       -1.6588818378
H                -3.4999145432       -2.3115662220       -1.1626220411
H                 4.5299446796        1.4455049080       -3.0046939753
H                 6.8342223371        1.6694897239       -3.3230886048
H                 7.7257434810        3.1268387329       -2.8216189649
H                 5.9997090723        3.2477736567       -3.2435480322


