%chk=./calculation/no_transfer_pubchem_sampling_0_b3lyp/131989682.chk
%nprocshared=7
%mem=6GB
#p b3lyp/6-31g(d) opt SCF=XQC 

Gaussian input prepared by ASE

0 1
C                 2.1829883993        6.5625129233        0.2775041913
C                 2.5580973865        5.3329809536        0.5767085445
C                 3.6172195721        4.5289984881       -0.1183204186
C                 3.9251884864        3.3860036190        0.8678570612
C                 3.7465871595        2.0096854186        0.2110214173
C                 4.0406650184        0.8534498100        1.1681238970
C                 4.6346322350       -0.3677128819        0.4466798814
C                 6.0650941697       -0.0873521930       -0.0877717069
C                 6.0213515780       -0.4912547210       -1.5676767470
O                 7.1672159967       -1.2207792028       -1.9250317744
C                 7.4064811062       -1.2820512186       -3.3105335650
C                 4.6752554550       -1.2635833626       -1.6689534033
O                 3.8609338268       -0.6441133511       -0.6970307782
C                 4.8400307500       -2.7831951257       -1.4468590222
C                 5.5790880216       -3.1266973486       -0.1507553005
C                 4.7376064163       -2.7750667256        1.0772296503
O                 4.6005513395       -1.4022854948        1.4039173493
C                 3.4007330350       -3.5367294438        0.9022006500
O                 2.5565436227       -3.2657650340        1.9936579936
C                 1.2156510866       -3.7250600897        1.8599864018
C                 1.0454483122       -4.7328844524        0.7222060218
C                 1.5090917816       -4.1590809749       -0.6203625513
C                 2.7173527826       -3.2298573960       -0.4357950918
O                 3.6004305806       -3.4597882099       -1.5278204264
C                 0.2628099500       -2.5259545914        1.7188018650
C                -1.1785962532       -2.8984100967        1.9577135655
O                -1.5188316864       -3.9770298326        2.3800592797
C                -2.1771968850       -1.8119914916        1.6163967608
C                -2.5509074675       -1.8671683468        0.1335067591
C                -3.1864897733       -0.5612491579       -0.3686988695
C                -2.1877856627        0.5710944352       -0.6089248531
C                -2.6297394496        1.8894405452        0.0233393507
O                -1.4814538799        2.7219534341        0.0974851404
C                -1.7182521273        3.8760723593        0.8779445386
C                -0.4071898153        4.6589942783        0.9822782982
C                 0.7466145587        3.7400782469        1.3692209962
C                 2.0478518675        4.4877216979        1.7125158418
O                 3.0897565325        3.5696607998        2.0013543582
C                -2.8465986081        4.7165437624        0.2729610853
C                -4.1224216021        3.8734580855        0.1125079000
C                -5.2511645120        4.6937933883       -0.4988457890
C                -3.7495799486        2.6360013089       -0.6702935313
C                -4.2662993247        2.2868258330       -1.8348289877
O                -3.8157625523       -0.9296189830       -1.5838267163
C                -4.2978125741       -2.2594893524       -1.4921064621
C                -5.8307564492       -2.3359634379       -1.4989846308
C                -6.5379905589       -1.4167579575       -0.4910818845
O                -7.8139221135       -1.9343395560       -0.1400772438
C                -6.7966506890       -0.0209360008       -1.0659309341
O                -7.3653361146        0.8313265211       -0.1044036484
C                -3.6327651587       -2.9081206845       -0.2388089295
O                -4.5710571012       -3.1215249791        0.7906115299
C                -4.5411626754       -4.4066362243        1.3766139619
H                 2.6128587210        7.0947857241       -0.5540312612
H                 1.4292614966        7.0806796338        0.8433867882
H                 3.2332733138        4.1338636040       -1.0613448344
H                 4.4983128646        5.1311277063       -0.3367012258
H                 4.9565498971        3.4554477286        1.2416357458
H                 4.4259803831        1.9673174228       -0.6427341830
H                 2.7320667823        1.9115963499       -0.1743149236
H                 3.1306673504        0.5354460459        1.6727928322
H                 4.7505150239        1.1572165841        1.9367821940
H                 6.3188841045        0.9633014584        0.0309076584
H                 6.8033083333       -0.6745654075        0.4525587221
H                 5.9466702491        0.4072133247       -2.2009037119
H                 8.3335275304       -1.8381139666       -3.4333047734
H                 6.5989646221       -1.7958646958       -3.8425697706
H                 7.5244504614       -0.2770960892       -3.7330212577
H                 4.1677797440       -1.1121932796       -2.6274576456
H                 5.4108482543       -3.1800954698       -2.2945391600
H                 6.5523650557       -2.6441867153       -0.1383797974
H                 5.7406817348       -4.2070320460       -0.1367654781
H                 5.2125945131       -3.1818527307        1.9799006886
H                 3.6785425783       -4.6069570268        0.8960516450
H                 0.9723981526       -4.2209195741        2.8106907587
H                -0.0014954451       -5.0298737302        0.6723025132
H                 1.6254345274       -5.6242399836        0.9661507816
H                 1.8136016682       -4.9676331387       -1.2860933291
H                 0.6997871232       -3.6179371131       -1.1117625490
H                 2.3977008137       -2.1816574195       -0.4278729985
H                 0.3679956612       -2.0460755580        0.7456619588
H                 0.5468022708       -1.7850097234        2.4714345877
H                -3.0852758785       -1.9475878618        2.2066245053
H                -1.7441719991       -0.8346896504        1.8305165036
H                -1.6664921069       -2.0685899621       -0.4796945636
H                -3.9500565064       -0.2383484902        0.3548652814
H                -2.0350065620        0.6977975361       -1.6789691134
H                -1.2216140486        0.3222611942       -0.1662759437
H                -2.9597400177        1.6762146002        1.0607701761
H                -2.0238280465        3.5525053952        1.8917711141
H                -0.5323681176        5.4463044649        1.7277210614
H                -0.1927810590        5.1250568430        0.0201065733
H                 0.9317586148        3.0509464289        0.5465045785
H                 0.4716152804        3.1464323088        2.2429257815
H                 1.8866539568        5.0851192785        2.6183126550
H                -2.5306813861        5.0713595396       -0.7103767135
H                -3.0406568965        5.5816733897        0.9092248113
H                -4.4425103658        3.5467922886        1.1114554861
H                -4.9710896357        5.0657984506       -1.4813816188
H                -5.4731790501        5.5465640018        0.1392394162
H                -6.1480257616        4.0867210288       -0.5886218230
H                -5.0443646457        2.8616701140       -2.3042093350
H                -3.9536556484        1.4011663742       -2.3580414120
H                -3.9390583491       -2.7803283576       -2.3916515239
H                -6.1844059967       -2.0902276296       -2.5017678268
H                -6.1109165353       -3.3707995107       -1.2889527975
H                -5.9315964381       -1.3210088707        0.4170837424
H                -7.6831174845       -2.7691845867        0.3205989204
H                -5.8594429399        0.4381056197       -1.3793751515
H                -7.4661459136       -0.1210041171       -1.9326050628
H                -8.1255577273        0.3629010940        0.2634939262
H                -3.1743319068       -3.8647305529       -0.5128575256
H                -3.5757970609       -4.6149709338        1.8449433739
H                -5.3192059597       -4.4049978576        2.1382319910
H                -4.7608296381       -5.1837614742        0.6352455136


