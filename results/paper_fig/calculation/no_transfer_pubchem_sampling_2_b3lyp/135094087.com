%chk=./calculation/no_transfer_pubchem_sampling_2_b3lyp/135094087.chk
%nprocshared=7
%mem=6GB
#p b3lyp/6-31g(d) opt SCF=XQC 

Gaussian input prepared by ASE

0 1
C                -6.4618223198        1.1782874424        1.2618529532
C                -5.2890338425        1.0079924768        0.3448573056
C                -5.4182184652        1.2239703248       -1.0212729686
C                -4.3341288250        1.0447073408       -1.8630126755
C                -3.1131859717        0.6486439347       -1.3536499972
C                -2.9710093443        0.4281953670        0.0149493196
N                -1.7167352874        0.0338644476        0.4844901462
C                -1.3216958699       -0.2898794636        1.7389191182
O                -2.0088437816       -0.2722782000        2.7363913580
C                 0.1624276118       -0.6654301885        1.7792103116
N                 0.6004558645       -1.2427388957        0.5217070223
C                 0.1259686867       -2.6134060836        0.3545606580
C                 1.1123711097       -3.6417469048        0.9067708448
C                 2.4311446329       -3.5702685466        0.1247064108
C                 2.5702646710       -2.2180884619       -0.5814756527
C                 2.0395480698       -1.0818725753        0.2910264352
C                 2.2694056855        0.2843073511       -0.3747901444
C                 3.7171355386        0.7706020948       -0.2590348540
N                 3.8579807909        2.0917940812       -0.8038197642
C                 4.1740738836        2.4344572442       -2.0687545072
C                 4.1239655358        3.8070702482       -2.1360667568
C                 3.7548311172        4.2047127522       -0.8441250687
N                 3.5961779836        3.1589407636       -0.0568475345
C                -4.0618929029        0.6101198065        0.8565375354
H                -7.1617973750        1.9109583266        0.8668680118
H                -6.9894410765        0.2299506520        1.3685855393
H                -6.1358493387        1.4929708953        2.2500105497
H                -6.3694872148        1.5350986837       -1.4276827312
H                -4.4410559023        1.2165601117       -2.9241299831
H                -2.2651654791        0.5107438043       -2.0097821736
H                -0.9912363665       -0.0885760840       -0.2157067866
H                 0.7139468076        0.2605072097        1.9660850616
H                 0.3242535113       -1.3251720227        2.6446603473
H                -0.8379397822       -2.7017418919        0.8594478729
H                -0.0407532139       -2.8000518910       -0.7109585466
H                 1.2878797270       -3.4230667745        1.9619703538
H                 0.6794382204       -4.6400983005        0.8467713061
H                 2.4739401121       -4.3655456998       -0.6208341294
H                 3.2709120173       -3.7148913682        0.8065560387
H                 3.6197596609       -2.0373958772       -0.8150871410
H                 2.0189551996       -2.2251542406       -1.5227681150
H                 2.5873807632       -1.1022394527        1.2489046315
H                 1.6287999827        1.0302856613        0.0989561709
H                 1.9850335258        0.2241012144       -1.4275766376
H                 4.3985042165        0.1010009280       -0.7893496534
H                 4.0055408000        0.8161272803        0.7944476893
H                 4.4149853762        1.6980983331       -2.8092616991
H                 4.3244307252        4.4306465026       -2.9826990469
H                 3.6058353608        5.1958111721       -0.4646867739
H                -3.9460558292        0.4391144724        1.9140813496


