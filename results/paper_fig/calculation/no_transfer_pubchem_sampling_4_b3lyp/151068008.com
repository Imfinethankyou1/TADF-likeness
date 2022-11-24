%chk=./calculation/no_transfer_pubchem_sampling_4_b3lyp/151068008.chk
%nprocshared=7
%mem=6GB
#p b3lyp/6-31g(d) opt SCF=XQC 

Gaussian input prepared by ASE

0 1
C                -8.2628124078       -0.4850931887        6.0730696434
C                -7.5489098924       -0.5370642259        4.7574200200
C                -6.4858720830       -1.4071027014        4.5549859524
C                -5.8340621317       -1.4747075507        3.3368990854
C                -6.2345816242       -0.6477868646        2.2940899108
N                -5.6129658605       -0.6725831485        1.0387226686
N                -6.3535204995       -0.4475978797       -0.0539026218
C                -5.5449695068       -0.5201454319       -1.0909369994
C                -6.0164089534       -0.3405587302       -2.5037431077
C                -7.4894977314        0.0679969831       -2.5101261891
C                -5.8418459407       -1.6679241161       -3.2521122062
C                -5.1729221230        0.7441851286       -3.1845450342
C                -4.2312355745       -0.7902290855       -0.6572669480
C                -4.3062338107       -0.8837934946        0.7141472370
C                -3.2063712577       -1.0455868349        1.7080817792
N                -1.9465418145       -0.6283212902        1.1367653809
C                -0.7936029044       -1.2925394081        1.4025745526
O                -0.7244344547       -2.2197336074        2.1760899163
N                 0.3154154513       -0.7937641658        0.7061644420
C                 1.6148873021       -1.0014665144        1.2133770893
C                 1.8526316897       -0.9610980769        2.5817582255
C                 3.1295150769       -1.1884212066        3.0580944736
C                 4.1745235872       -1.4362887452        2.1859877407
C                 3.9526171495       -1.4636217043        0.8150437032
C                 5.0785258648       -1.7085554463       -0.1490575542
C                 5.7179789625       -0.3956111939       -0.6347339044
C                 6.4297127866        0.3305916527        0.5128479345
C                 7.3238948028        1.4644649427       -0.0051922282
C                 6.5701747296        2.4357059161       -0.9270632716
C                 6.7059945305        1.8123048457       -2.3302694751
C                 7.5512218381        0.5490685864       -2.1035893735
C                 6.6922265562       -0.6800708524       -1.7851108272
N                 8.2928693293        0.8878997865       -0.9093386578
C                 9.5506873183        0.5085066578       -0.6603906277
O                10.2753004305       -0.0568039109       -1.4460549121
O                 9.9599701563        0.8562715589        0.5893589717
C                 2.6672234938       -1.2509884921        0.3399293331
C                 0.1624535752       -0.1890078750       -0.5461656218
C                -0.6351345075       -0.7735731551       -1.5242633802
C                -0.7799650655       -0.1489493589       -2.7489438336
F                -1.5692397695       -0.7232486657       -3.6798563886
C                -0.1469626450        1.0406905103       -3.0457351248
C                 0.6681231404        1.6217498930       -2.0827656060
C                 0.8264483859        1.0158284200       -0.8398395957
O                 1.6069085702        1.5749733723        0.1346118724
C                 2.4724632994        2.5728037342       -0.3381218779
O                 1.8380178914        3.5125706591       -1.1328237080
C                 1.3674586320        2.9314959779       -2.3350210860
C                -7.3128588505        0.2136872891        2.4783469284
C                -7.9586088328        0.2636082473        3.6962547965
H                -9.1497567899       -1.1186004364        6.0398744813
H                -7.6236328408       -0.8405491941        6.8774721153
H                -8.5852874369        0.5290437215        6.2983465884
H                -6.1646097197       -2.0523879721        5.3596792934
H                -5.0374393880       -2.1871503844        3.1981400924
H                -7.8471978223        0.1722057830       -3.5320678052
H                -7.6198764763        1.0145441113       -1.9926363319
H                -8.0872055931       -0.6804683784       -1.9982653735
H                -6.4172068689       -2.4533379973       -2.7671031104
H                -4.7957947610       -1.9644471736       -3.2693970312
H                -6.1879223455       -1.5669010656       -4.2783279651
H                -4.1305314396        0.4423371905       -3.2459209742
H                -5.2328074231        1.6780129978       -2.6289892924
H                -5.5372514699        0.9185942736       -4.1944344201
H                -3.3603841368       -0.9021202364       -1.2677053319
H                -3.0685555254       -2.0909918272        2.0079294276
H                -3.4342523422       -0.4593470233        2.6105930594
H                -1.9392446270        0.2069128848        0.5744785001
H                 1.0409292569       -0.7709305368        3.2644951107
H                 3.3114513067       -1.1680498724        4.1222339437
H                 5.1660646247       -1.6195169294        2.5721745424
H                 4.6993461845       -2.2508298470       -1.0181792874
H                 5.8459466416       -2.3235271375        0.3246841937
H                 4.9118345949        0.2453987461       -1.0059692086
H                 7.0667142502       -0.3752664705        1.0503240398
H                 5.6933460549        0.7249701615        1.2148513407
H                 7.8238213270        1.9653025442        0.8247757458
H                 7.0428016650        3.4166810270       -0.8985241915
H                 5.5348990725        2.5530337788       -0.6132794953
H                 5.7435650486        1.5655887760       -2.7746025644
H                 7.2232078896        2.4932970497       -3.0045562486
H                 8.2413229707        0.3338644888       -2.9211615708
H                 7.3635815472       -1.4946918402       -1.5049646166
H                 6.1408550841       -0.9901218706       -2.6745358233
H                10.8865779754        0.5962280698        0.6841380172
H                 2.4766070168       -1.2962725300       -0.7238584200
H                -1.1229833730       -1.7157062308       -1.3297916455
H                -0.2854647729        1.5080034893       -4.0068034898
H                 3.3011046677        2.0910809587       -0.8941038891
H                 2.8584278632        3.0888543838        0.5435377120
H                 0.6827335862        3.6611836166       -2.7719254659
H                 2.2087880246        2.7797252637       -3.0322236717
H                -7.6315856517        0.8269855076        1.6506486890
H                -8.7926201577        0.9372018900        3.8295188349


