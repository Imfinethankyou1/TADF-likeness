%chk=./calculation/no_transfer_pubchem_sampling_5_b3lyp/24905276.chk
%nprocshared=7
%mem=6GB
#p b3lyp/6-31g(d) opt SCF=XQC 

Gaussian input prepared by ASE

0 1
C                -5.2220006605        2.0454678569        0.0261681279
C                -3.9054588979        1.3296567941        0.3105367881
C                -4.0182744771        0.4654004748        1.5627331153
C                -3.5144500845        0.4721401261       -0.9024547274
N                -2.2223923064       -0.1293761671       -0.7365407923
N                -1.0973414469        0.5927156900       -0.8054834959
C                -0.0763492817       -0.2017074079       -0.5438726678
C                 1.2870698313        0.3076000658       -0.5200581243
C                 1.6607280701        1.3181998397       -1.4339074403
C                 2.9324562253        1.8095841939       -1.4434720838
C                 3.9052278946        1.3274428187       -0.5404482176
C                 5.2299642148        1.8081584385       -0.5321864349
C                 6.1422533469        1.3280271043        0.3659266149
C                 5.7705626385        0.3457983573        1.2973712955
C                 4.4940627024       -0.1418529583        1.3143432623
C                 3.5300399969        0.3310298180        0.3987568577
C                 2.2093007253       -0.1581031490        0.3885443818
C                -0.5473103162       -1.5259005072       -0.3014085905
C                -0.0746320395       -2.8385655994       -0.0950533056
N                 1.2341857952       -3.1654484478       -0.0270223413
N                -0.9529379074       -3.8288397287        0.0588198250
C                -2.2494675215       -3.5646326885       -0.0282142056
N                -2.8287832453       -2.4149208156       -0.2897151294
C                -1.9546335049       -1.4129605434       -0.4282795561
H                -5.5013553309        2.6709606955        0.8702212387
H                -6.0167902014        1.3221244983       -0.1427278317
H                -5.1365346960        2.6770945979       -0.8550002125
H                -3.1159474369        2.0740092408        0.4602557110
H                -3.0695657220       -0.0160663117        1.7882125261
H                -4.7678829514       -0.3102499000        1.4219068657
H                -4.3023091560        1.0731156879        2.4185818530
H                -4.2277126744       -0.3442650043       -1.0362905629
H                -3.4895446208        1.0952460226       -1.8010677148
H                 0.9119175079        1.6857413834       -2.1180750751
H                 3.2161223999        2.5802437812       -2.1462211317
H                 5.5112468667        2.5658161893       -1.2498820294
H                 7.1556261681        1.7020155708        0.3668668605
H                 6.5034197642       -0.0210012528        2.0011755852
H                 4.2032495340       -0.8983283647        2.0298824237
H                 1.9203845039       -0.8836768976        1.1365408287
H                 1.4517650996       -4.1472414942       -0.0085755465
H                 1.9299557287       -2.5125714692       -0.3438411022
H                -2.9078635346       -4.4118805387        0.1129561587


