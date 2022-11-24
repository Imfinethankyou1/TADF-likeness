%chk=calculation/no_transfer_pubchem_sampling_2_b3lyp/44062256_rwf.chk
%rwf=calculation/no_transfer_pubchem_sampling_2_b3lyp/44062256_rwf.rwf
%nprocshared=14
%mem=3GB
#p b3lyp/6-31g(d) 6D 10F nosymm GFInput td=(50-50,nstate=2) SCF=XQC 

Gaussian input prepared by ASE

0 1
C                -5.8979984852        2.3729860426       -1.6524980070
C                -5.1392175950        1.2956910329       -0.9147917599
C                -4.7901836488        0.0978302881       -1.5568840851
C                -4.0709961792       -0.8938458168       -0.8943277900
C                -3.6894897230       -0.6910311952        0.4355601519
S                -2.7163917096       -1.9440364494        1.2850762907
O                -2.9344046333       -1.7519036817        2.7210108248
O                -2.9764337776       -3.2199218555        0.6143331432
N                -1.0990799729       -1.5859271478        1.0159671758
C                -0.5527689287       -0.4083806907        1.7180833339
C                 0.9199583439       -0.2419584478        1.3438737040
C                 1.0808030541        0.0708864755       -0.1720035370
C                 2.4740965862       -0.2282281131       -0.6242086324
N                 2.8915172971       -1.0352559008       -1.5476681891
N                 4.2848195502       -0.9559226326       -1.5582952898
C                 4.6014485211       -0.1002860580       -0.6301098341
C                 5.9184980460        0.3336205396       -0.2434643710
C                 7.1622358653       -0.0056070367       -0.7070420047
C                 8.0921850931        0.7510378670        0.0667122818
C                 7.3492143504        1.4909344653        0.9394983394
O                 6.0216121933        1.2505174101        0.7661338340
O                 3.5081824232        0.4074585254        0.0109039919
C                 0.0251459938       -0.6936425917       -1.0177015814
C                -0.4877216869       -1.9360439475       -0.2854068065
C                -4.0395082623        0.4854351688        1.1024396226
C                -4.7596749832        1.4676513471        0.4229255742
H                -6.6382426487        1.9444748464       -2.3365654999
H                -6.4205967838        3.0410208731       -0.9607504996
H                -5.2193370750        2.9906746730       -2.2556454019
H                -5.0978862991       -0.0661344021       -2.5869267216
H                -3.8348396351       -1.8312454046       -1.3867745071
H                -1.1047147631        0.5098732773        1.4553294692
H                -0.6635821173       -0.5660485331        2.7927089752
H                 1.3622379985        0.5594095906        1.9437634383
H                 1.4564053241       -1.1632043604        1.5983219469
H                 0.9241307023        1.1498820515       -0.3063758377
H                 7.3700848568       -0.7085914742       -1.5005149390
H                 9.1697478595        0.7440628306       -0.0162929481
H                 7.5958796556        2.2028655122        1.7121147900
H                 0.4545518766       -0.9891576113       -1.9786504979
H                -0.8195426484       -0.0276067038       -1.2242931246
H                 0.3362601497       -2.6380383642       -0.1083554882
H                -1.2255540644       -2.4721978892       -0.8841273221
H                -3.7791411104        0.6096593596        2.1481560079
H                -5.0417120099        2.3782471312        0.9459497803


