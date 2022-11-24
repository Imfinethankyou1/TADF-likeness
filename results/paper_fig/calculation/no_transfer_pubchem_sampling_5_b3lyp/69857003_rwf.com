%chk=calculation/no_transfer_pubchem_sampling_5_b3lyp/69857003_rwf.chk
%rwf=calculation/no_transfer_pubchem_sampling_5_b3lyp/69857003_rwf.rwf
%nprocshared=14
%mem=3GB
#p b3lyp/6-31g(d) 6D 10F nosymm GFInput td=(50-50,nstate=2) SCF=XQC 

Gaussian input prepared by ASE

0 1
C                -4.0351099006        5.3833102006       -1.3903295927
C                -2.8590069368        4.5027216890       -1.0429835093
C                -1.8871124773        4.9543917742       -0.1078067108
C                -0.8022456388        4.1767283160        0.2229691202
C                -0.6129810132        2.8951807654       -0.3614931190
C                 0.5145004063        2.0695442500       -0.0837006301
O                 1.5377519095        2.5077503607        0.7209871235
C                 0.6291850597        0.7925430780       -0.6178768125
O                 1.7427951294        0.0162132310       -0.3978516512
C                 2.0229822694       -0.3475234973        0.9673856717
C                 3.1327648229       -1.3716348927        0.9863045111
C                 3.2109364093       -2.2894731040        2.0417606869
C                 4.2579076216       -3.2098073577        2.1090194490
C                 5.2363483844       -3.2293980355        1.1136173615
C                 5.1611203725       -2.3216971755        0.0554351140
C                 4.1191532521       -1.3953236076       -0.0072406730
C                -0.3777232913        0.3160968934       -1.5136282285
O                -0.2847514244       -0.8735544556       -2.1885818309
C                 0.0353556382       -2.1007766052       -1.5086978457
C                -0.9934521584       -2.5246842754       -0.4813388257
C                -2.3635321809       -2.3520773699       -0.7175180565
C                -3.3035658338       -2.7877588507        0.2153154466
C                -2.8878449779       -3.4125766536        1.3939571277
C                -1.5258938108       -3.5953436996        1.6342985024
C                -0.5849896135       -3.1484832080        0.7035522098
C                -1.4469222625        1.1250168415       -1.8430828918
C                -1.5938339008        2.4154071591       -1.2857240140
C                -2.6982469221        3.2529736403       -1.6031022911
H                -4.7031842856        4.8945024798       -2.1060961092
H                -3.7064342098        6.3324736999       -1.8328862629
H                -4.6238263602        5.6340384168       -0.4986000452
H                -2.0135451274        5.9278772600        0.3603116225
H                -0.1072784065        4.5474917893        0.9742603907
H                 1.5106910239        3.4755601030        0.7575167769
H                 1.1149366651       -0.7541781051        1.4315008899
H                 2.3146036480        0.5480156342        1.5262428010
H                 2.4484328730       -2.2819420612        2.8182305743
H                 4.3038920351       -3.9156828959        2.9342595551
H                 6.0491707225       -3.9492266972        1.1603803903
H                 5.9174001239       -2.3329428703       -0.7252645301
H                 4.0571755153       -0.6922460182       -0.8312770645
H                 0.0697474297       -2.8272852347       -2.3282346420
H                 1.0308910458       -2.0466246844       -1.0658971299
H                -2.6874829483       -1.8687074372       -1.6344314233
H                -4.3632963894       -2.6424728026        0.0217460126
H                -3.6216666671       -3.7532448186        2.1196504901
H                -1.1932391012       -4.0798287868        2.5487813074
H                 0.4763178993       -3.2912099328        0.8958523638
H                -2.1815669967        0.7474734645       -2.5470903162
H                -3.4353274221        2.8803930863       -2.3116102927


