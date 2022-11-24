%chk=./calculation/no_transfer_pubchem_sampling_4_b3lyp/34263779.chk
%nprocshared=7
%mem=6GB
#p b3lyp/6-31g(d) opt SCF=XQC 

Gaussian input prepared by ASE

0 1
C                 5.4750172101       -2.5240277776        1.6186589330
N                 5.5544370939       -1.3549400925        0.7810418685
C                 5.3765099002       -1.4002507354       -0.5522117379
O                 5.1595478871       -2.4134240572       -1.1788474522
C                 5.5089542581       -0.0371568465       -1.2444020484
O                 5.4711427023        1.0710935208       -0.3734136649
C                 4.2746813029        1.4500878835        0.1673001548
C                 4.3179238379        2.5776259714        0.9877538904
C                 3.1633198411        3.0325719982        1.5916612149
C                 1.9598379560        2.3845674273        1.3843523317
C                 1.9147451054        1.2633161872        0.5603511870
N                 0.6774043174        0.6420833126        0.3734958565
C                 0.2983830897       -0.1831666377       -0.6521553799
O                 1.0485553239       -0.6511372788       -1.4805756362
C                -1.1460563142       -0.4871583577       -0.6316984078
S                -1.6687171790       -2.0658564885       -1.0988159980
C                -3.3183085152       -1.5751969541       -0.7450378657
C                -4.4377018434       -2.4750688923       -0.8992345329
C                -4.2566279646       -3.7893303661       -1.3297380701
C                -5.3432764783       -4.6300986311       -1.4687100056
C                -6.6198586622       -4.1725816375       -1.1815208308
C                -6.8076452176       -2.8670203923       -0.7541231815
C                -5.7270258158       -2.0199246407       -0.6125247297
N                -3.4210405034       -0.3508182394       -0.3415796761
C                -2.2345542652        0.2908429857       -0.2826073258
C                -2.2182443841        1.7055229076        0.0949333465
C                -1.3769177181        2.6173413014       -0.5419211170
C                -1.3801668593        3.9464041262       -0.1625145039
C                -2.2259055851        4.3820882507        0.8450674588
C                -3.0829232330        3.4858829223        1.4630491107
C                -3.0844398000        2.1552046467        1.0890234909
C                 3.0746338341        0.7831766237       -0.0392103735
H                 5.2453610122       -3.3705833138        0.9733132393
H                 4.6854731491       -2.4105414934        2.3640128237
H                 6.4249405399       -2.7023673889        2.1261600236
H                 5.7647044344       -0.4687963557        1.2142045551
H                 6.4957592612        0.0013119131       -1.7168896243
H                 4.7510686737        0.0317882061       -2.0323809896
H                 5.2622391683        3.0771684088        1.1327906914
H                 3.1994728099        3.9042190302        2.2272344295
H                 1.0540714063        2.7464377210        1.8493767021
H                -0.0635874420        1.0059904073        0.9575255494
H                -3.2609896221       -4.1440787049       -1.5542134445
H                -5.1958392470       -5.6462524432       -1.8028614057
H                -7.4671952717       -4.8327600995       -1.2919311986
H                -7.8023969704       -2.5101834536       -0.5318485692
H                -5.8538711652       -1.0005519022       -0.2841428745
H                -0.7316868983        2.2849169054       -1.3422457947
H                -0.7227790877        4.6448553889       -0.6583552926
H                -2.2235839837        5.4206672879        1.1405065533
H                -3.7529003433        3.8258535770        2.2388495547
H                -3.7491769368        1.4478415541        1.5607166402
H                 3.0152311918       -0.0955892844       -0.6575268745


