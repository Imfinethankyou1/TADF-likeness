%chk=calculation/no_transfer_pubchem_sampling_2_b3lyp/126266507_rwf.chk
%rwf=calculation/no_transfer_pubchem_sampling_2_b3lyp/126266507_rwf.rwf
%nprocshared=14
%mem=3GB
#p b3lyp/6-31g(d) 6D 10F nosymm GFInput td=(50-50,nstate=2) SCF=XQC 

Gaussian input prepared by ASE

0 1
C                 5.8141307895       -3.5766587857        0.2934176133
C                 6.1829166375       -2.6110869186       -0.8275374741
O                 6.1247543209       -1.2408005552       -0.4133144232
C                 4.9301663635       -0.5904146092       -0.3644340314
C                 4.9903912435        0.7384705483        0.0866573493
C                 3.8361535804        1.5059476238        0.1555546477
C                 2.5963070164        0.9633251156       -0.2113559007
N                 1.4007553061        1.7594600208       -0.0929514906
C                 0.7995792480        1.9047954783        1.2643879301
C                 0.2522331283        3.2982596975        1.6058531269
C                -0.2051641325        0.7852294412        1.6486966275
O                -0.0225362160        0.1785536252        2.6977832967
N                -1.2549808545        0.5893137166        0.7913341356
C                -2.3009254302       -0.3601057448        0.9126266625
C                -2.1621167907       -1.4811956637        1.7424929153
C                -3.1899684087       -2.4155679402        1.8241367347
C                -4.3484396378       -2.2492034176        1.0787093943
C                -4.4945535012       -1.1347346381        0.2359915809
C                -5.7394041189       -1.0443443310       -0.5881674433
O                -5.8778200991       -0.4268338936       -1.6273463463
O                -6.7506534803       -1.7756855436       -0.0520785180
C                -7.9632126524       -1.7904061479       -0.8175946515
C                -3.4780006834       -0.1552265406        0.1510371107
C                -3.6091451999        1.1057628273       -0.6742348101
S                 0.7351483758        2.3911792596       -1.4984004546
C                 1.5223614298        3.9973788198       -1.7426115155
O                -0.6913528555        2.6457828176       -1.2198195771
O                 1.1304489286        1.5463134409       -2.6228451089
C                 2.5377422204       -0.3591001797       -0.6533578024
C                 3.6928844086       -1.1364215845       -0.7347532036
H                 5.9392494426       -4.6106223101       -0.0481630292
H                 4.7772690759       -3.4472356044        0.6181858358
H                 6.4651313222       -3.4179963991        1.1589039572
H                 5.5610764666       -2.7629113730       -1.7186780629
H                 7.2249998856       -2.7549957921       -1.1263021844
H                 5.9557048169        1.1457589481        0.3694380841
H                 3.8892525817        2.5339426571        0.5049078621
H                 1.6358201028        1.7114155932        1.9382593475
H                 1.0383373797        4.0538296760        1.4960933455
H                -0.6013548101        3.5789171572        0.9870725466
H                -0.0642200485        3.2941338495        2.6538057145
H                -1.2920926608        1.2141952437       -0.0107161244
H                -1.2588305238       -1.5979362188        2.3238690276
H                -3.0765548674       -3.2812606544        2.4705819105
H                -5.1471963028       -2.9784384605        1.1322701021
H                -8.3646639658       -0.7788967701       -0.9229695744
H                -8.6543602854       -2.4201671755       -0.2560382780
H                -7.7877538094       -2.2048748668       -1.8142456426
H                -4.6292702445        1.2599615393       -1.0150830738
H                -3.2930449476        1.9799908899       -0.0936728209
H                -2.9763772639        1.0728718658       -1.5703616991
H                 1.2764788822        4.6499948079       -0.9039656786
H                 1.1311457977        4.4085368818       -2.6761674607
H                 2.5999150344        3.8411971444       -1.8210181973
H                 1.5813333226       -0.7808680218       -0.9427083914
H                 3.6123056831       -2.1597535458       -1.0811738897


