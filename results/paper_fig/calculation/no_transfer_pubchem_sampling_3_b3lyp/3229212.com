%chk=./calculation/no_transfer_pubchem_sampling_3_b3lyp/3229212.chk
%nprocshared=7
%mem=6GB
#p b3lyp/6-31g(d) opt SCF=XQC 

Gaussian input prepared by ASE

0 1
C                -8.6307589539       -1.3849601860       -1.0849182510
C                -7.3834967291       -0.8033640363       -0.4984745709
C                -6.6018667723       -1.5585587575        0.3725625878
C                -5.4378180744       -1.0403622435        0.9090872736
C                -5.0605797689        0.2431780230        0.5661677918
S                -3.5409262035        0.9136603839        1.2132375667
O                -3.5984229999        2.3480732613        1.2419056960
O                -3.1693462179        0.2184523262        2.4124722903
C                -2.2612800484        0.4672095163       -0.0821652552
N                -1.7381842409       -0.7935946781        0.2738852432
N                -2.5013465478       -1.8917175091        0.1349220695
N                -1.8803085996       -2.8618547479        0.6521945494
N                -0.7186530675       -2.4769862888        1.1558251420
C                -0.6167003076       -1.1831545815        0.9443802118
C                 0.6389275525       -0.4069814027        1.1908115418
C                 0.3961640323        1.0604173944        1.4519868460
C                -0.4585437680        1.4453104219        2.4832531013
C                -0.6972972389        2.7918233295        2.6914934409
C                -0.0626901797        3.7073768525        1.8639230186
N                 0.7887241078        3.3546915725        0.9167646108
C                 1.0212565593        2.0678962484        0.7257146784
N                 1.5807763464       -0.6444837918        0.1055162896
C                 1.0536692963       -0.5351303217       -1.2415161684
C                 2.2368702768       -0.2551258143       -2.1901769311
N                 3.4276131865       -0.9324530655       -1.7187934565
C                 4.5780809032       -0.2294461577       -1.4547213396
C                 4.9360691846        0.8682216581       -2.2542399152
C                 6.1080985590        1.5615027507       -2.0407472720
C                 6.9861512335        1.1878962826       -1.0360808822
C                 6.6497387802        0.0960665370       -0.2519881185
C                 7.5597678482       -0.3777173523        0.8423037196
F                 6.9160583160       -0.5318316941        2.0165058148
F                 8.0994149450       -1.5846752860        0.5662645363
F                 8.5895049609        0.4438688482        1.0830298013
C                 5.4705789511       -0.6058820054       -0.4458057868
C                 3.1067679499       -2.1593120763       -1.0310899125
C                 2.3356426650       -1.8796601039        0.2691735676
C                -5.8145568097        1.0160418374       -0.2948054863
C                -6.9790721050        0.4884730527       -0.8233133215
H                -9.1778324113       -1.9550392625       -0.3377615829
H                -9.2773484958       -0.6071954813       -1.4820464354
H                -8.3737732799       -2.0640626206       -1.8978646919
H                -6.9095790057       -2.5601624497        0.6316921476
H                -4.8254728070       -1.6092912291        1.5887962922
H                -1.5090321553        1.2513600993       -0.0777118257
H                -2.7835358828        0.4090327571       -1.0360563107
H                 1.0900652478       -0.8373574571        2.0943309905
H                -0.9379831822        0.7005782658        3.1004813406
H                -1.3642520856        3.1180872888        3.4719317887
H                -0.2297641299        4.7706206040        1.9685354909
H                 1.7509874373        1.8304294376       -0.0353482875
H                 0.5216582031       -1.4499979807       -1.5542253261
H                 0.3512004132        0.2977113064       -1.2978638738
H                 2.4214017723        0.8189696889       -2.2153203339
H                 1.9822416151       -0.5920203723       -3.2037127159
H                 4.2977968740        1.1666043716       -3.0710565840
H                 6.3508806905        2.4009086434       -2.6742022274
H                 7.9049877573        1.7248910963       -0.8677429782
H                 5.2576905109       -1.4334723909        0.2102155029
H                 4.0114325093       -2.7322259864       -0.8345361634
H                 2.4869989096       -2.7526251571       -1.7114773492
H                 1.6799917391       -2.7328455748        0.5041546314
H                 3.0244834954       -1.7276475216        1.1046328659
H                -5.4899552035        2.0197809313       -0.5201400142
H                -7.5813205577        1.0850707979       -1.4922470718


