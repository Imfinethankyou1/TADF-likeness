%chk=calculation/no_transfer_pubchem_sampling_4_b3lyp/10555723_rwf.chk
%rwf=calculation/no_transfer_pubchem_sampling_4_b3lyp/10555723_rwf.rwf
%nprocshared=14
%mem=3GB
#p b3lyp/6-31g(d) 6D 10F nosymm GFInput td=(50-50,nstate=2) SCF=XQC 

Gaussian input prepared by ASE

0 1
C                 0.8557458245       -2.8630757892       -1.9246630254
C                 0.1878384763       -2.4869270087       -0.5983589814
C                -0.3232445572       -3.6996310636        0.1664521471
O                -0.9081667343       -1.5937891619       -0.8043854381
C                -0.3538463359       -0.2798533675       -0.8034706790
C                -1.4677740934        0.8126682754       -0.6323456131
O                -2.7739178661        0.2629584511       -0.6344356092
C                -3.2686355524       -0.0855192882       -1.9341786145
C                -4.6014344244       -0.7723861232       -1.7763964973
C                -5.7839975325       -0.1595921969       -2.2002674758
C                -7.0006741502       -0.8265762641       -2.0390567975
C                -7.0480433588       -2.0885984337       -1.4554279774
C                -5.8653719265       -2.7135609510       -1.0198562109
N                -5.8985357481       -4.0071624703       -0.4863958294
C                -4.6465948790       -2.0415018274       -1.1915066455
C                -1.4381666771        1.7971575711        0.5791987455
C                -2.2481504372        1.3861203695        1.8420874993
C                -1.8337739890        0.1921123069        2.6778119620
C                -0.9652018052        0.3397664247        3.7698166073
C                -0.6444710905       -0.7469677750        4.5851030401
C                -1.2022742423       -2.0008809442        4.3321426200
C                -2.0752460333       -2.1588818769        3.2537028863
C                -2.3828878550       -1.0746453234        2.4299720878
S                 0.1992539425        2.5216169608        1.1409091928
C                 1.5455171081        2.0499321755       -0.0363293878
C                 1.3747563995        2.5825616612       -1.4830722416
C                 1.4037710390        4.0933223137       -1.5818478838
C                 2.6227210152        4.7835827383       -1.5198917296
C                 2.6667172209        6.1751290816       -1.6029658869
C                 1.4859297939        6.9043760066       -1.7568363943
C                 0.2661165820        6.2306336521       -1.8269753460
C                 0.2288120832        4.8381293496       -1.7395899016
C                 1.9419521878        0.5394154140        0.0852668883
O                 2.6796981791        0.0964049113       -1.0489848525
C                 4.0717734444        0.4024763101       -1.0442264017
C                 4.8912180393       -0.3284382863        0.0053597212
C                 6.0866803906        0.2367491960        0.4670527191
C                 6.8699229798       -0.4662481116        1.3831733868
C                 6.4665084580       -1.7114027805        1.8596392934
C                 5.2599476802       -2.2773793224        1.4149344114
N                 4.8525701152       -3.5416316281        1.8548065149
C                 4.4863917911       -1.5779559541        0.4743503405
C                 0.7327626956       -0.3893933427        0.2728748949
O                 1.1222052436       -1.7655676415        0.2196841926
H                 1.7001126639       -3.5349378225       -1.7397073096
H                 0.1407784797       -3.3719255505       -2.5797595157
H                 1.2424620553       -1.9746722448       -2.4297071771
H                 0.5027832481       -4.3873557666        0.3734395211
H                -0.7650612409       -3.3766036949        1.1113345487
H                -1.0753643621       -4.2322731425       -0.4251180258
H                 0.1338766610       -0.1031234833       -1.7719025489
H                -1.3753853589        1.4570876837       -1.5183248579
H                -3.3719458312        0.8261805914       -2.5433130578
H                -2.5550010723       -0.7537401167       -2.4310266279
H                -5.7543729908        0.8291923526       -2.6501325520
H                -7.9241991266       -0.3563676291       -2.3671541722
H                -8.0001129982       -2.6017907450       -1.3376761161
H                -6.7774129375       -4.2631999571       -0.0530617754
H                -5.1156639886       -4.2343197465        0.1145699832
H                -3.7178611056       -2.5070653051       -0.8688413854
H                -1.9813007661        2.6690880183        0.2008149556
H                -3.2733915284        1.2349165099        1.4921883411
H                -2.2683119048        2.2737563420        2.4873496746
H                -0.5421829733        1.3171386741        3.9846273952
H                 0.0305882256       -0.6091798678        5.4259533411
H                -0.9660523849       -2.8454347658        4.9744321415
H                -2.5260747781       -3.1291261070        3.0581504295
H                -3.0556191762       -1.2006237165        1.5869509086
H                 2.3667390216        2.6284535103        0.4047340957
H                 2.1903183893        2.1619365432       -2.0798363261
H                 0.4530947597        2.2036709904       -1.9317447043
H                 3.5501826623        4.2235340507       -1.4127873143
H                 3.6232136744        6.6893420772       -1.5554754321
H                 1.5175163279        7.9884296539       -1.8259665926
H                -0.6586235651        6.7883927318       -1.9503556659
H                -0.7267351838        4.3218655792       -1.7975280431
H                 2.5616856183        0.4359016537        0.9845145506
H                 4.4103697266        0.1175527888       -2.0476130671
H                 4.2374359514        1.4874260099       -0.9487768824
H                 6.4027116471        1.2156624974        0.1139940525
H                 7.7998085889       -0.0337066726        1.7440736242
H                 7.0795319121       -2.2468121644        2.5815340426
H                 5.2142254515       -3.8101383961        2.7619935094
H                 3.8527397003       -3.6963126598        1.7988830344
H                 3.5428980945       -1.9931860229        0.1330944553
H                 0.3091970121       -0.1951819178        1.2603038139


