%chk=./calculation/no_transfer_pubchem_sampling_6_b3lyp/1403061.chk
%nprocshared=7
%mem=6GB
#p b3lyp/6-31g(d) opt SCF=XQC 

Gaussian input prepared by ASE

0 1
C                 7.0227469358       -1.5153751901        2.4686112174
C                 6.8895157756       -1.4481706743        0.9456351215
C                 5.4471204497       -1.3183099749        0.5488670613
C                 4.8467375386       -0.0710158946        0.4351408160
C                 3.5099719244        0.0638020377        0.1124367017
C                 2.7337939223       -1.0707629480       -0.1040976859
N                 1.3844943438       -1.0282016508       -0.4500194714
C                 0.5408390937        0.0373799543       -0.5592673127
N                 0.9433171508        1.2483890641       -0.2771871767
C                 0.0519757538        2.2670949060       -0.3405980929
C                 0.4622885203        3.5763373451       -0.0429443351
C                -0.4527218416        4.5970923101       -0.0694301648
C                -1.7940616771        4.3497915588       -0.3879389927
C                -2.2195994373        3.0810533415       -0.6895577050
C                -1.3001355203        2.0231231940       -0.6793799187
N                -1.6997389538        0.7696939581       -1.0172041424
C                -0.8272581659       -0.1915861669       -0.9889160871
N                -1.3006993117       -1.4318209075       -1.3513139619
S                -2.7007904135       -1.4533324609       -2.3486098523
O                -2.4553665328       -0.6857325374       -3.5297505007
O                -2.9547717249       -2.8560622061       -2.5033337021
C                -3.9472914865       -0.7059645512       -1.3269808338
C                -4.6461464946        0.3853955290       -1.7992225268
C                -5.6219617448        0.9420041885       -0.9899656778
C                -5.8780586962        0.4025879808        0.2601425398
C                -5.1768059722       -0.7047452226        0.7264647230
C                -5.4487117268       -1.2710389780        2.0855704804
C                -4.2015077336       -1.2629752523       -0.0926609103
C                 3.3278361335       -2.3278172034        0.0082137059
C                 4.6640537595       -2.4453223271        0.3321748604
H                 6.4780476959       -2.3725224561        2.8581893293
H                 8.0675230957       -1.6071150669        2.7565317468
H                 6.6138736918       -0.6158592386        2.9231492942
H                 7.4539297689       -0.5909301817        0.5734508646
H                 7.3145870270       -2.3534771275        0.5074630995
H                 5.4387808717        0.8179945263        0.5990388411
H                 3.0542605104        1.0346320380        0.0261831460
H                 0.9642276396       -1.9385472749       -0.5701175494
H                 1.4974492260        3.7486896507        0.2076153443
H                -0.1416027422        5.6057657259        0.1598796465
H                -2.4951065673        5.1713999001       -0.3971100932
H                -3.2458724064        2.8645792247       -0.9412265401
H                -0.6442891386       -2.1411721036       -1.6385010611
H                -4.4037120060        0.7788425749       -2.7724315972
H                -6.1852431234        1.7973700944       -1.3321379655
H                -6.6386345502        0.8463266336        0.8864690689
H                -4.8103281645       -0.7840924709        2.8231571161
H                -5.2382930958       -2.3370496145        2.1091130635
H                -6.4834650117       -1.1061367821        2.3754735105
H                -3.6376635234       -2.1284913024        0.2185118281
H                 2.7360277461       -3.2178314675       -0.1614611310
H                 5.1064381881       -3.4278855043        0.4138838617


