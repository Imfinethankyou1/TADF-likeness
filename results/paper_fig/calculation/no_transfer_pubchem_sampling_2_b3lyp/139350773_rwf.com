%chk=calculation/no_transfer_pubchem_sampling_2_b3lyp/139350773_rwf.chk
%rwf=calculation/no_transfer_pubchem_sampling_2_b3lyp/139350773_rwf.rwf
%nprocshared=14
%mem=3GB
#p b3lyp/6-31g(d) 6D 10F nosymm GFInput td=(50-50,nstate=2) SCF=XQC 

Gaussian input prepared by ASE

0 1
C                 5.3852659756        2.3697988685        1.1362367415
N                 5.1177317606        1.8969900209       -0.2212512678
C                 4.3165082615        2.5123070718       -1.1449261673
N                 4.2599435507        1.8325395478       -2.2636417772
N                 5.0609402082        0.7216715037       -2.0904092983
C                 5.5587613206        0.7736687362       -0.8730057746
C                 6.4055212834       -0.3250181725       -0.2948297805
F                 7.0398224244        0.1382770301        0.8664159395
C                 5.6081332634       -1.5999182027        0.0542183988
C                 4.4434505599       -1.3063548441        0.9707200876
C                 4.6052294254       -1.2964249616        2.3602567121
C                 3.5198893636       -0.9732220218        3.1762818254
C                 2.2773180939       -0.6460617222        2.6394832014
C                 2.1052377560       -0.6427627823        1.2416509588
N                 0.8732278270       -0.3053762060        0.6406217014
C                 0.7060121891       -0.3187031265       -0.8207656773
C                -0.7297432214        0.0932311574       -1.0057763164
C                -1.3050643954        0.3293631482        0.2365175527
C                -2.6305834798        0.7272354132        0.3807974243
C                -3.4077212532        0.8916546665       -0.7706809119
C                -4.8476671811        1.3702853577       -0.6767655117
N                -5.5279772272        0.8283815089        0.4935125784
C                -6.8607701605        1.3974390897        0.8040907169
C                -7.7907760467        0.8611813349       -0.3223678099
C                -7.1769897477       -0.5452067180       -0.5978149255
C                -5.9902156253       -0.5808261376        0.4086951186
C                -6.5536473429       -0.7622621504        1.8339110345
C                -7.1782793656        0.6435821429        2.1132159118
C                -2.8301561181        0.6424986707       -2.0262381354
C                -1.4957737443        0.2471564878       -2.1607101060
C                -0.8928470206       -0.0293703118       -3.5106836515
F                -0.4708886311       -1.3150633734       -3.6048358617
F                 0.1909566158        0.7480641478       -3.7348448687
F                -1.7616554141        0.1832863239       -4.5188634530
C                -0.3000202717        0.0821560960        1.3013566640
O                -0.4680366160        0.1958200494        2.5060300695
C                 3.1962100460       -0.9831371818        0.4239195824
C                 5.3737563754       -2.4523480908       -1.2226777053
O                 6.5342613089       -3.2635533367       -0.9284109476
C                 6.5871490279       -2.7466384520        0.4174691667
H                 4.8115454262        3.2853993724        1.2917673554
H                 5.0752703859        1.6207962555        1.8664940089
H                 6.4480357339        2.5804849132        1.2647122329
H                 3.8140515596        3.4495224939       -0.9482304562
H                 7.1905724688       -0.5897804058       -1.0115832868
H                 5.5693597936       -1.5356093769        2.7996915319
H                 3.6393931588       -0.9740767378        4.2565791228
H                 1.4444639833       -0.3943239078        3.2794741616
H                 1.4004588390        0.3783791206       -1.3066686138
H                 0.9004135098       -1.3177280004       -1.2298838308
H                -3.0602306178        0.8943069907        1.3627692536
H                -4.8484626896        2.4641687359       -0.5683724279
H                -5.3547983526        1.1597852504       -1.6348986979
H                -6.8370813739        2.4866214003        0.8986897649
H                -8.8380315188        0.8182071033       -0.0064444702
H                -7.7471667998        1.5010903422       -1.2104719237
H                -7.8834586289       -1.3626681361       -0.4225776809
H                -6.8291173979       -0.6389463387       -1.6321482846
H                -5.1860605844       -1.2715937129        0.1449054173
H                -7.2844619905       -1.5754710097        1.8874397161
H                -5.7485134311       -0.9833732792        2.5402124935
H                -8.2536570402        0.6033063472        2.3141557041
H                -6.6923428294        1.1290661501        2.9642258925
H                -3.4338044511        0.7565952020       -2.9212120085
H                 3.0927683608       -0.9778032987       -0.6547141422
H                 5.4452155602       -1.9437006191       -2.1885419488
H                 4.4459894047       -3.0381987117       -1.1772282434
H                 6.1705731440       -3.4545069112        1.1467337651
H                 7.6025626023       -2.4642918136        0.7171671563


